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

class BasicConv1DExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        ts_shape = observation_space.spaces["time_series"].shape # VD: (96, 23)
        port_shape = observation_space.spaces["portfolio_features"].shape # VD: (4,)
        
        seq_len = ts_shape[0]          # 96 nến
        num_ts_features = ts_shape[1]  # 23 features
        port_input_dim = port_shape[0] # 4 features

        # --- NHÁNH 1: TIME SERIES (1D CNN) ---
        # Input đi vào sẽ được permute thành (batch_size, 23, 96)
        self.cnn = nn.Sequential(
            # Lớp 1: Dùng 64 filter trượt qua chuỗi, bắt mẫu hình ngắn hạn
            nn.Conv1d(in_channels=num_ts_features, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2), # Chiều dài nén từ 96 -> 48
            
            # Lớp 2: Dùng 128 filter, bắt mẫu hình phức tạp hơn
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2), # Chiều dài nén từ 48 -> 24
            
            nn.Flatten() # Duỗi phẳng ma trận thành vector 1 chiều
        )

        # Tính toán chiều của vector sau khi duỗi: 128 (channels) * 24 (seq_len còn lại)
        cnn_out_dim = 128 * (seq_len // 4) # = 3072

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
        # Gộp CNN (128) + Portfolio (16) = 144
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