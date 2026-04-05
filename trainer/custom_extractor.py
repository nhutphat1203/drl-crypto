import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
 
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.Linear(hidden_dim, num_heads)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x shape: (batch_size, window_size, hidden_dim)
        
        # 1. Tính điểm attention cho từng time step trên mỗi head
        # scores shape: (batch_size, window_size, num_heads)
        scores = self.attention(x)
        
        # 2. Áp dụng Softmax dọc theo chiều thời gian (window_size)
        # weights shape: (batch_size, window_size, num_heads)
        weights = th.softmax(scores, dim=1)
        
        # 3. Nhân có trọng số (Context Vector)
        # Sử dụng einsum để tính toán nhanh ma trận đa chiều
        # b: batch, s: window_size, h: heads, d: hidden_dim
        # context shape: (batch_size, num_heads, hidden_dim)
        context = th.einsum("bsh,bsd->bhd", weights, x)
        
        # 4. Flatten các heads lại thành một vector biểu diễn duy nhất
        # Output shape: (batch_size, num_heads * hidden_dim)
        return context.reshape(context.shape[0], -1)


class GRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)       
        
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        window_size = ts_shape[0] 
        num_ts_features = ts_shape[1]  
        port_input_dim = port_shape[0]
        
        print(f"GRUExtractor - Time Series Shape: {ts_shape}, Portfolio Shape: {port_shape}")
        print(f"Window Size: {window_size}")
        
        self.gru_hidden_dim = 64
        
        # 1. GRU Layer
        self.gru = nn.GRU(
            input_size=num_ts_features,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )       
        
        # SỬA LỖI 1: Chiều sau khi Flatten phải nhân với window_size
        self.gru_flatten_dim = window_size * self.gru_hidden_dim
        mlp_summary_out_dim = 512
        
        # SỬA LỖI 2: Đưa gru_flatten_dim vào Linear
        self.mlp_summary = nn.Sequential(
            nn.LayerNorm(self.gru_flatten_dim),
            nn.Linear(self.gru_flatten_dim, mlp_summary_out_dim),
            nn.GELU(),
        )
        
        concat_dim = mlp_summary_out_dim + port_input_dim
        
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
        
        # 1. Trích xuất đặc trưng thời gian với GRU
        gru_out, _ = self.gru(ts_data)
        
        # 2. Trải phẳng (Flatten) thay vì dùng Attention
        gru_out_flat = gru_out.reshape(gru_out.shape[0], -1)
        
        # 3. Đưa qua MLP để tóm tắt
        ts_features = self.mlp_summary(gru_out_flat)
        
        # 4. Gộp với Portfolio và xuất ra
        combined = th.cat([ts_features, port_data], dim=1) 
        return self.fusion_mlp(combined)

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Lấy thông số từ không gian quan sát
        ts_shape = observation_space.spaces["time_series"].shape # (window_size, features)
        port_shape = observation_space.spaces["portfolio_features"].shape # (4,)
        
        window_size = ts_shape[0] 
        num_ts_features = ts_shape[1]  
        port_input_dim = port_shape[0]
        
        # --- CẤU HÌNH SIÊU THAM SỐ ---
        self.lstm_hidden_dim = 64 # Giữ 64 để tương đương với sức mạnh của GRU bạn đang dùng
        mlp_summary_out_dim = 512
        
        # 1. Nhánh LSTM
        self.lstm = nn.LSTM(
            input_size=num_ts_features,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        ) 
        
        # 2. Khối tóm tắt chuỗi (Flatten All Hidden States)
        # Chiều sau khi Flatten = window_size * lstm_hidden_dim
        self.lstm_flatten_dim = window_size * self.lstm_hidden_dim
        
        self.mlp_summary = nn.Sequential(
            nn.LayerNorm(self.lstm_flatten_dim),
            nn.Linear(self.lstm_flatten_dim, mlp_summary_out_dim),
            nn.GELU(),
        )
        
        # 3. Khối Fusion (Kết hợp TS Features + Portfolio Features)
        concat_dim = mlp_summary_out_dim + port_input_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim)
        )

        self._initialize_weights()
        
    def _initialize_weights(self):
        # Khởi tạo trọng số cho LSTM (Khác một chút so với GRU do có nhiều cổng hơn)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Kỹ thuật chuyên gia: Để forget gate bias ban đầu cao (1.0) 
                # giúp mạng dễ học các phụ thuộc dài hạn ngay từ đầu
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0) # Forget gate bias

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]
        
        # 1. Chạy qua LSTM 
        # LSTM trả về: (output, (h_n, c_n))
        # output chứa toàn bộ hidden states của tất cả các timesteps
        lstm_out, _ = self.lstm(ts_data)
        
        # 2. Trải phẳng (Flatten) toàn bộ hidden states (Batch, Window * Hidden)
        lstm_out_flat = lstm_out.reshape(lstm_out.shape[0], -1)
        
        # 3. Đưa qua MLP để nén/tóm tắt đặc trưng thời gian
        ts_features = self.mlp_summary(lstm_out_flat)
        
        # 4. Gộp với Portfolio và xuất ra Features cuối cùng
        combined = th.cat([ts_features, port_data], dim=1) 
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