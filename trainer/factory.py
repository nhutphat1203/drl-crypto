from typing import List
import pandas as pd
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.wrappers import ActionMasker
from .trainer import Trainer
import os
from dataclasses import dataclass
from environment.market import Market
from config import Config
from stable_baselines3.common.utils import set_random_seed

@dataclass
class DataMetadata:
    data_train: pd.DataFrame
    data_val: pd.DataFrame

def get_trainer(data_metadata: List[DataMetadata],
                config: Config) -> Trainer:
    
    seed = config.model_env.seed
    set_random_seed(seed)

    def make_env(data_df: pd.DataFrame, rank: int, test_mode: bool = False, verbose: int = 0):
        def _init():
            prefix = "Test" if test_mode else "Train"
            env = Market(
                df=data_df,
                name=f"{prefix}_env_{rank}",
                initial_balance=config.model_env.initial_balance,
                window_size=config.model_env.window_size,
                episode_length=config.model_env.tick_per_episode,
                test_mode=test_mode,
                verbose=verbose
            )
            
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            
            if not test_mode:
                log_sub_path = os.path.join(config.settings.folder_path, config.settings.log_path, f"env_{rank}")
                os.makedirs(log_sub_path, exist_ok=True)
                env = Monitor(env, log_sub_path)
            return env
        return _init

    train_env_fns = [make_env(meta.data_train, i, verbose=config.settings.train_verbose) for i, meta in enumerate(data_metadata)]
    train_venv = DummyVecEnv(train_env_fns)
    train_venv.seed(seed)

    train_venv = VecNormalize(
        train_venv, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.,
        gamma=config.parameters.gamma
    )

    val_env_fns = [make_env(meta.data_val, i, test_mode=True, verbose=config.settings.eval_verbose) for i, meta in enumerate(data_metadata)]
    val_venv = DummyVecEnv(val_env_fns)
    val_venv.seed(seed)
    val_venv = VecNormalize(
        val_venv, 
        norm_obs=True, 
        norm_reward=False,
        clip_obs=10.
    )

    val_venv.obs_rms = train_venv.obs_rms 
    val_venv.training = False

    trainer = Trainer(
        train_env=train_venv,
        val_env=val_venv,
        config=config,
    )

    return trainer