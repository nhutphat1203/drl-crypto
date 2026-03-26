import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
