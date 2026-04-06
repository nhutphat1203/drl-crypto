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
        
        # 3. Cấu hình LSTM (Để 128 cho giống GRU của bạn)
        self.lstm_hidden_dim = 128
        
        self.lstm = nn.LSTM(
            input_size=num_ts_features,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )       
        
        # 4. Attention Mechanism (Dùng lại class TemporalAttention bạn đã viết)
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
        
        ts_shape = observation_space.spaces["time_series"].shape # (12, 23)
        port_shape = observation_space.spaces["portfolio_features"].shape # (4,)
        
        window_size = ts_shape[0] 
        num_ts_features = ts_shape[1]
        port_input_dim = port_shape[0]

        # --- NHÁNH 1: TIME SERIES (1D-CNN) ---
        self.cnn_out_channels = 64
        
        self.cnn = nn.Sequential(
            # Lớp 1: Kernel 3 để bắt các mẫu hình nến ngắn hạn (3 nến liên tiếp)
            nn.Conv1d(in_channels=num_ts_features, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(32), # Giúp ổn định giá trị sau khi Conv
            
            # Lớp 2: Dilation=2 giúp tăng tầm nhìn lên tương đương 5 nến nhưng ít tham số hơn
            nn.Conv1d(in_channels=32, out_channels=self.cnn_out_channels, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(self.cnn_out_channels),
            
            # Lớp 3: Pointwise Conv để nén thông tin channel
            nn.Conv1d(in_channels=self.cnn_out_channels, out_channels=self.cnn_out_channels, kernel_size=1),
            nn.GELU(),
        )

        # Tính toán chiều sau khi CNN (vì padding/dilation làm thay đổi kích thước seq)
        # Với kernel 3, padding 1 -> seq vẫn là 12. 
        # Với kernel 3, padding 2, dilation 2 -> seq vẫn là 12.
        self.cnn_flatten_dim = self.cnn_out_channels * window_size # 64 * 12 = 768
        
        # --- NHÁNH 2: FUSION MLP ---
        # concat_dim = 768 (CNN) + 23 (Residual) + 4 (Port) = 795
        concat_dim = self.cnn_flatten_dim + num_ts_features + port_input_dim
        
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
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"] # (Batch, 12, 23)
        port_data = observations["portfolio_features"] # (Batch, 4)

        # 1. Chuẩn bị cho Conv1d: (Batch, Features, Seq)
        ts_data_cnn = ts_data.transpose(1, 2)
        
        # 2. Chạy qua CNN: (Batch, 64, 12)
        cnn_feat = self.cnn(ts_data_cnn)
        
        # 3. Làm phẳng toàn bộ đặc trưng: (Batch, 768)
        cnn_flat = th.flatten(cnn_feat, start_dim=1)
        
        # 4. Nhánh Residual (Nến hiện tại): (Batch, 23)
        last_bar = ts_data[:, -1, :]
        
        # 5. Kết hợp (CNN + Residual + Portfolio)
        combined = th.cat([cnn_flat, last_bar, port_data], dim=1) 
        
        return self.fusion_mlp(combined)