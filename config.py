from dataclasses import dataclass
import yaml

@dataclass
class SettingsConfig:
    folder_path: str
    log_path: str
    model_save_path: str
    checkpoint_path: str
    checkpoint_freq: int
    checkpoint_name_pfx: str
    best_model_save_path: str
    eval_freq: int
    tensorboard_log: str
    device: str

@dataclass
class ModelEnvConfig:
    window_size: int
    initial_balance: float
    tick_per_episode: int
    seed: int

@dataclass
class ParametersConfig:
    total_timesteps: int
    learning_rate: float
    gamma: float
    n_steps: int
    ent_coef: float
    clip_range: float
    n_epochs: int
    batch_size: int
    gae_lambda: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float

@dataclass
class Config:
    settings: SettingsConfig
    model_env: ModelEnvConfig
    parameters: ParametersConfig

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            settings=SettingsConfig(**cfg['settings']),
            model_env=ModelEnvConfig(**cfg['model_env']),
            parameters=ParametersConfig(**cfg['parameters'])
        )

def load_config(path: str) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return Config.from_dict(cfg)
