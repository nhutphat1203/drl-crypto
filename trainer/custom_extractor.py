import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class BasicGRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        ts_shape = observation_space.spaces["time_series"].shape #23*96
        port_shape = observation_space.spaces["portfolio_features"].shape #4
        
        num_ts_features = ts_shape[1]  # 23 features
        port_input_dim = port_shape[0]

        # --- NHÁNH 1: TIME SERIES (GRU) ---
        self.gru_hidden_dim = 96
        self.gru = nn.GRU(
            input_size=num_ts_features,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # --- NHÁNH 2: PORTFOLIO (MLP) ---
        self.port_out_dim = 16
        self.port_hidden_dim = 32
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(port_input_dim, self.port_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.port_hidden_dim, self.port_out_dim),
            nn.SiLU()
        )

        # --- NHÁNH 3: FUSION MLP ---
        concat_dim = self.gru_hidden_dim + self.port_out_dim
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.SiLU() # Kích hoạt trước khi đẩy vào Policy Network
        )

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]

        # Chạy qua GRU
        gru_out, _ = self.gru(ts_data)
        
        # CHỈ LẤY LAST STATE (Nến cuối cùng)
        ts_features = gru_out[:, -1, :]               

        # Chạy qua Portfolio MLP
        port_features = self.portfolio_mlp(port_data)   

        # Gộp (Concat) và Fusion
        combined = th.cat([ts_features, port_features], dim=1) 
        return self.fusion_mlp(combined)          

class AdvancedGRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        ts_shape = observation_space.spaces["time_series"].shape
        port_shape = observation_space.spaces["portfolio_features"].shape
        
        num_ts_features = ts_shape[1]
        port_input_dim = port_shape[0]

        # --- NHÁNH 1: TIME SERIES (GRU) ---
        self.gru_hidden_dim = 96
        self.gru = nn.GRU(
            input_size=num_ts_features,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # --- NHÁNH 2: PORTFOLIO (MLP) ---
        self.port_out_dim = 16
        self.port_hidden_dim = 32
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(port_input_dim, self.port_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.port_hidden_dim, self.port_out_dim),
            nn.SiLU()
        )

        # --- NHÁNH 3: FUSION MLP ---
        # Đã nâng cấp: Đầu ra của chuỗi thời gian giờ gồm Last + Max + Mean 
        # nên kích thước sẽ gấp 3 lần gru_hidden_dim
        self.ts_feature_dim = self.gru_hidden_dim * 3 
        concat_dim = self.ts_feature_dim + self.port_out_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.SiLU() # Kích hoạt trước khi đẩy vào Policy Network
        )

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]

        # 1. Chạy qua GRU
        # gru_out có shape: (batch_size, seq_len, gru_hidden_dim)
        gru_out, _ = self.gru(ts_data)
        
        # 2. TRÍCH XUẤT ĐẶC TRƯNG TỪ GRU (Last + Max + Mean)
        # 2.a. Last state (Nến cuối cùng)
        last_state = gru_out[:, -1, :] 
        
        # 2.b. Max Pooling (Đặc trưng nổi bật nhất trong toàn chuỗi)
        # th.max trả về (values, indices), ta chỉ lấy values ở index 0
        max_pool = th.max(gru_out, dim=1)[0] 
        
        # 2.c. Mean Pooling (Đặc trưng trung bình của toàn chuỗi)
        mean_pool = th.mean(gru_out, dim=1)
        
        # Gộp 3 đặc trưng này lại. Shape mới: (batch_size, gru_hidden_dim * 3)
        ts_features = th.cat([last_state, max_pool, mean_pool], dim=1)

        # 3. Chạy qua Portfolio MLP
        port_features = self.portfolio_mlp(port_data)   

        # 4. Gộp (Concat) tất cả và Fusion
        combined = th.cat([ts_features, port_features], dim=1) 
        return self.fusion_mlp(combined) 

class BasicConv1DExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        ts_shape = observation_space.spaces["time_series"].shape # Hiện tại là (48, 23)
        port_shape = observation_space.spaces["portfolio_features"].shape # (4,)
        
        seq_len = ts_shape[0]          # 48 nến
        num_ts_features = ts_shape[1]  # 23 features
        port_input_dim = port_shape[0] # 4 features

        # --- NHÁNH 1: TIME SERIES  ---
        self.cnn = nn.Sequential(
            # Lớp 1: Giảm xuống 32 filter để chạy siêu tốc
            nn.Conv1d(in_channels=num_ts_features, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2), # Chiều dài chuỗi nén: 48 -> 24
            # Lớp 2: Giảm xuống 64 filter
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2), # Chiều dài chuỗi nén: 24 -> 12
            nn.Flatten() 
        )

        # Tính toán tự động kích thước sau khi duỗi: 
        # 64 channels * 12 nhịp thời gian còn lại = 768
        cnn_out_dim = 64 * (seq_len // 4) 

        self.cnn_compressor = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.SiLU()
        )

        # --- NHÁNH 2: PORTFOLIO (MLP) ---
        self.port_out_dim = 16
        self.port_hidden_dim = 32
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(port_input_dim, self.port_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.port_hidden_dim, self.port_out_dim),
            nn.SiLU()
        )

        # --- NHÁNH 3: FUSION MLP ---
        concat_dim = 128 + self.port_out_dim 
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, features_dim),
            nn.SiLU() 
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, observations: dict) -> th.Tensor:
        ts_data = observations["time_series"]
        port_data = observations["portfolio_features"]

        ts_data = ts_data.permute(0, 2, 1)

        cnn_features = self.cnn(ts_data)
        cnn_features = self.cnn_compressor(cnn_features) 
        
        port_features = self.portfolio_mlp(port_data)   
        combined = th.cat([cnn_features, port_features], dim=1) 
        return self.fusion_mlp(combined)

