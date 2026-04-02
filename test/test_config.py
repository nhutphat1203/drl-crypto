import pytest
import os
import tempfile
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Config, load_config, SettingsConfig, ModelEnvConfig, ParametersConfig

@pytest.fixture
def valid_yaml_dict():
    return {
        "settings": {
            "folder_path": "te1",
            "log_path": "logs/",
            "model_save_path": "models/ppo",
            "checkpoint_path": "checkpoints/",
            "checkpoint_freq": 100,
            "checkpoint_name_pfx": "chkpt",
            "best_model_save_path": "best",
            "eval_freq": 10,
            "tensorboard_log": "tb/",
            "device": "cpu"
        },
        "model_env": {
            "window_size": 10,
            "initial_balance": 1000.0,
            "tick_per_episode": 32,
            "seed": 42
        },
        "parameters": {
            "total_timesteps": 1000,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "n_steps": 128,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "n_epochs": 10,
            "batch_size": 64,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.01
        }
    }

def test_config_from_dict(valid_yaml_dict):
    config = Config.from_dict(valid_yaml_dict)
    
    assert isinstance(config.settings, SettingsConfig)
    
    assert isinstance(config.model_env, ModelEnvConfig)
    assert config.model_env.seed == 42
    
    assert isinstance(config.parameters, ParametersConfig)
    assert config.parameters.learning_rate == 0.001

def test_load_config_from_file(valid_yaml_dict):
    # Setup temporary yaml file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_yaml_dict, f)
        temp_path = f.name
        
    try:
        config = load_config(temp_path)
        assert config.settings.checkpoint_freq == 100
        assert config.parameters.gamma == 0.99
    finally:
        os.remove(temp_path)

def test_missing_config_fields(valid_yaml_dict):
    # Remove a required field
    del valid_yaml_dict["settings"]["folder_path"]
    
    # Dataclass raises TypeError when required fields are missing
    with pytest.raises(TypeError):
        Config.from_dict(valid_yaml_dict)
