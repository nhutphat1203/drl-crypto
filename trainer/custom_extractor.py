import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
 
class GRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)       
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        num_ts_features = ts_shape[1]
        port_input_dim = port_shape[0]
        # 1. Nhánh Time Series
        self.gru_hidden_dim = 42
        self.gru_output_dim = self.gru_hidden_dim * 3  # (Last + Max + Mean)
        self.gru = nn.GRU(
            input_size=num_ts_features,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.ts_norm = nn.LayerNorm(self.gru_output_dim)
        # 2. Khối Fusion 
        # Nối: (42 * 3) + 4 = 130 chiều
        concat_dim = self.gru_output_dim + port_input_dim 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.GELU(),
        )
        # Khởi tạo trọng số
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
        # 1. Chạy qua GRU
        gru_out, _ = self.gru(ts_data)
        # 2. Trích xuất (Last + Max + Mean)
        last_state = gru_out[:, -1, :] 
        max_pool = th.max(gru_out, dim=1)[0] 
        mean_pool = th.mean(gru_out, dim=1)
        ts_features = th.cat([last_state, max_pool, mean_pool], dim=1)
        layer_normed_ts = self.ts_norm(ts_features)
        # 3. Gộp (Concat) trực tiếp với port_data (ĐÃ SỬA LỖI GỌI HÀM)
        combined = th.cat([layer_normed_ts, port_data], dim=1) 
        # 4. Fusion
        return self.fusion_mlp(combined)

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        num_ts_features = ts_shape[1]
        port_input_dim = port_shape[0]

        # --- NHÁNH 1: TIME SERIES (LSTM) ---
        self.lstm_hidden_dim = 56
        self.lstm = nn.LSTM(
            input_size=num_ts_features,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # --- NHÁNH 2: PORTFOLIO (MLP) ---
        self.port_out_dim = 16
        self.port_hidden_dim = 32
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(port_input_dim, self.port_hidden_dim),
            nn.GELU(),
            nn.Linear(self.port_hidden_dim, self.port_out_dim),
        )

        # --- NHÁNH 3: FUSION MLP ---
        self.ts_feature_dim = self.lstm_hidden_dim * 3 
        concat_dim = self.ts_feature_dim + self.port_out_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
        )

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]

        # 1. Chạy qua LSTM
        # Lấy output (toàn bộ chuỗi hidden states), bỏ qua tuple (h_n, c_n)
        lstm_out, _ = self.lstm(ts_data)
        
        # 2. TRÍCH XUẤT ĐẶC TRƯNG (Last + Max + Mean)
        last_state = lstm_out[:, -1, :] 
        max_pool = th.max(lstm_out, dim=1)[0] 
        mean_pool = th.mean(lstm_out, dim=1)
        
        ts_features = th.cat([last_state, max_pool, mean_pool], dim=1)

        # 3. Chạy qua Portfolio MLP
        port_features = self.portfolio_mlp(port_data)   

        # 4. Gộp và Fusion
        combined = th.cat([ts_features, port_features], dim=1) 
        return self.fusion_mlp(combined)

class CNN1DExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        num_ts_features = ts_shape[1]
        port_input_dim = port_shape[0]

        # --- NHÁNH 1: TIME SERIES (1D-CNN) ---
        self.cnn_out_channels = 48
        
        # Thiết kế khối CNN với 2 lớp Conv1d
        self.cnn = nn.Sequential(
            # Lớp 1: Trích xuất các mẫu hình nến cơ bản 
            nn.Conv1d(in_channels=num_ts_features, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            # Lớp 2: Kết hợp các mẫu hình cơ bản thành các xu hướng phức tạp hơn
            nn.Conv1d(in_channels=32, out_channels=self.cnn_out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

        # --- NHÁNH 2: PORTFOLIO (MLP) ---
        self.port_out_dim = 16
        self.port_hidden_dim = 32
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(port_input_dim, self.port_hidden_dim),
            nn.GELU(),
            nn.Linear(self.port_hidden_dim, self.port_out_dim),
        )

        # --- NHÁNH 3: FUSION MLP ---
        self.ts_feature_dim = self.cnn_out_channels * 3 
        concat_dim = self.ts_feature_dim + self.port_out_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
        )

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]

        # 1. Chuyển đổi chiều Tensor cho Conv1d
        # Từ (Batch, Seq, Features) -> (Batch, Features, Seq)
        ts_data_cnn = ts_data.transpose(1, 2)
        
        # Chạy qua mạng CNN. Output shape: (Batch, cnn_out_channels, Seq)
        cnn_out = self.cnn(ts_data_cnn)
        
        # Chuyển đổi lại chiều Tensor để Pooling nhất quán với cấu trúc cũ
        # Từ (Batch, Channels, Seq) -> (Batch, Seq, Channels)
        cnn_out = cnn_out.transpose(1, 2)
        
        # 2. TRÍCH XUẤT ĐẶC TRƯNG (Last + Max + Mean)
        # last_state ở đây đại diện cho đặc trưng được tổng hợp tại cây nến cuối cùng
        last_state = cnn_out[:, -1, :] 
        max_pool = th.max(cnn_out, dim=1)[0] 
        mean_pool = th.mean(cnn_out, dim=1)
        
        ts_features = th.cat([last_state, max_pool, mean_pool], dim=1)

        # 3. Chạy qua Portfolio MLP
        port_features = self.portfolio_mlp(port_data)   

        # 4. Gộp và Fusion
        combined = th.cat([ts_features, port_features], dim=1) 
        return self.fusion_mlp(combined)