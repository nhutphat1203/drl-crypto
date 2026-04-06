import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        attn_scores = self.attention(x) 
        attn_weights = th.softmax(attn_scores, dim=1)  
        context_vector = th.sum(attn_weights * x, dim=1)  
        return context_vector

class GRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)       
        
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        window_size = ts_shape[0] 
        num_ts_features = ts_shape[1]  
        port_input_dim = port_shape[0]
        
        self.input_norm = nn.LayerNorm(num_ts_features)
        
        self.gru_hidden_dim = 128
        
        # 1. GRU Layer
        self.gru = nn.GRU(
            input_size=num_ts_features,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )       
        
        self.attention = TemporalAttention(self.gru_hidden_dim)
        
        self.layer_norm = nn.LayerNorm(self.gru_hidden_dim)
        
        concat_dim = self.gru_hidden_dim + port_input_dim
        
        # Lớp gộp cuối cùng
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim)
        )

        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]
        
        ts_data = self.input_norm(ts_data)
            
        gru_out, _ = self.gru(ts_data)
        
        context_vector = self.attention(gru_out)
        context_vector_norm = self.layer_norm(context_vector)
        
        combined = th.cat([context_vector_norm, port_data], dim=1) 
        return self.fusion_mlp(combined)

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)       
        
        # 1. Lấy thông số từ không gian quan sát
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        num_ts_features = ts_shape[1]  
        port_input_dim = port_shape[0]
        
        # 2. Normalization đầu vào
        self.input_norm = nn.LayerNorm(num_ts_features)
        
        # 3. Cấu hình LSTM
        self.lstm_hidden_dim = 128
        
        self.lstm = nn.LSTM(
            input_size=num_ts_features,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )       
        
        # 4. Attention Mechanism 
        self.attention = TemporalAttention(self.lstm_hidden_dim)
        
        # 5. LayerNorm sau Attention
        self.layer_norm = nn.LayerNorm(self.lstm_hidden_dim)
        
        # 6. Fusion MLP (Kết hợp TS Features + Portfolio Features)
        concat_dim = self.lstm_hidden_dim + port_input_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim)
        )

        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Kỹ thuật cho LSTM: Khởi tạo forget gate bias = 1
                nn.init.zeros_(param.data)
                n = param.size(0)
                # Trong PyTorch LSTM bias sắp xếp theo: [b_ii, b_if, b_ig, b_io]
                # Ta set b_if (forget gate) thành 1
                param.data[n//4:n//2].fill_(1.0)

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]
        
        # Bước 1: Chuẩn hóa đầu vào
        ts_data = self.input_norm(ts_data)
            
        # Bước 2: Qua LSTM (Lấy toàn bộ sequence output)
        # LSTM trả về (output, (h_n, c_n)) -> ta chỉ lấy output
        lstm_out, _ = self.lstm(ts_data)
        
        # Bước 3: Qua Attention để tóm tắt chuỗi thành 1 context vector
        context_vector = self.attention(lstm_out)
        context_vector_norm = self.layer_norm(context_vector)
        
        # Bước 4: Concatenate với dữ liệu portfolio
        combined = th.cat([context_vector_norm, port_data], dim=1) 
        
        # Bước 5: Fusion qua MLP cuối cùng
        return self.fusion_mlp(combined)

class CNN1DExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)       
        
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        window_size = ts_shape[0] 
        num_ts_features = ts_shape[1]  
        port_input_dim = port_shape[0]
        
        self.cnn_hidden_dim = 128
        
        # 1. Khối CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_ts_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2), # Giảm seq_len từ 24 xuống 12
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=64, out_channels=self.cnn_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.cnn_hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2), # Giảm seq_len từ 12 xuống 6
            # Output shape lúc này: (batch_size, 128, 6)
        )
        
        # 2. Lớp Temporal Attention
        self.attention = TemporalAttention(self.cnn_hidden_dim)
        
        self.layer_norm = nn.LayerNorm(self.cnn_hidden_dim)
        
        concat_dim = self.cnn_hidden_dim + port_input_dim
        
        # 3. Lớp gộp cuối cùng
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim)
        )

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]
        
        # Permute cho Conv1d: (Batch, Seq_Len, Features) -> (Batch, Features, Seq_Len)
        ts_data = ts_data.permute(0, 2, 1) 
            
        cnn_out = self.cnn(ts_data) # Shape: (batch_size, 128, 6)
        
        # Permute ngược lại cho Temporal Attention: (Batch, Channels, Seq_Len) -> (Batch, Seq_Len, Channels)
        cnn_out_permuted = cnn_out.permute(0, 2, 1) # Shape: (batch_size, 6, 128)
        
        # Đưa qua Attention
        context_vector = self.attention(cnn_out_permuted) # Shape: (batch_size, 128)
        context_vector_norm = self.layer_norm(context_vector)
        
        combined = th.cat([context_vector_norm, port_data], dim=1) 
        return self.fusion_mlp(combined)