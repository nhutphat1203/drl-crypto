import pytest
import pandas as pd
import numpy as np
import sys
import os
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Config
from trainer.factory import get_trainer, DataMetadata
from trainer.trainer import Trainer

@pytest.fixture
def dummy_config():
    cfg = {
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
            "window_size": 5,
            "initial_balance": 1000.0,
            "tick_per_episode": 20,
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
    return Config.from_dict(cfg)

@pytest.fixture
def dummy_metadata():
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.ones(50),
        'high': np.ones(50),
        'low': np.ones(50),
        'close': np.ones(50),
        'volume': np.ones(50),
        'feature1': np.random.rand(50)
    })
    return [DataMetadata(data_train=df.copy(), data_val=df.copy())]

def test_get_trainer_initializes_vec_envs(dummy_metadata, dummy_config):
    trainer = get_trainer(dummy_metadata, dummy_config)
    
    assert isinstance(trainer, Trainer)
    assert trainer.config == dummy_config
    
    # Verify train_env
    assert isinstance(trainer.train_env, VecNormalize)
    assert trainer.train_env.norm_obs is True
    assert trainer.train_env.norm_reward is True
    assert trainer.train_env.gamma == dummy_config.parameters.gamma
    
    # Verify val_env
    assert isinstance(trainer.val_env, VecNormalize)
    assert trainer.val_env.norm_obs is True
    assert trainer.val_env.norm_reward is False  # Evaluation shouldn't normalize rewards dynamically
    
    # Verify obs_rms are shared
    assert trainer.train_env.obs_rms is trainer.val_env.obs_rms
    assert trainer.val_env.training is False
