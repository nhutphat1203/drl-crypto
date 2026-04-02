import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList, 
    StopTrainingOnNoModelImprovement 
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable
from config import Config
import os
from dataclasses import dataclass, field
from trainer.custom_extractor import GRUExtractor, LSTMExtractor, CNN1DExtractor

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Hàm giảm dần Learning Rate.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

@dataclass
class Trainer:
    train_env: gym.Env
    val_env: gym.Env
    config: Config
    folder_path: str
    extractor_type: str
        
    def train(self):
        config = self.config
        train_env = self.train_env
        val_env = self.val_env
        early_stopping_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=30,
            min_evals=12,              
            verbose=1
        )
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=os.path.join(self.folder_path, config.settings.best_model_save_path),
            log_path=os.path.join(self.folder_path, config.settings.log_path),
            eval_freq=config.settings.eval_freq,
            deterministic=True,
            render=False,
            verbose=1,
            n_eval_episodes=1,
            callback_after_eval=early_stopping_callback
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=config.settings.checkpoint_freq,
            save_path=os.path.join(self.folder_path, config.settings.checkpoint_path),
            name_prefix=config.settings.checkpoint_name_pfx
        )

        callback = CallbackList([checkpoint_callback, eval_callback])

        extractor_type = None
        if self.extractor_type == 'CNN':
            extractor_type = CNN1DExtractor
        elif self.extractor_type == 'GRU':
            extractor_type = GRUExtractor
        elif self.extractor_type == 'LSTM':
            extractor_type = LSTMExtractor

        policy_kwargs = dict(
            net_arch=dict(pi=[128], vf=[128]),
            activation_fn=th.nn.GELU,
            optimizer_class=th.optim.AdamW,
            optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=1e-4
            ),
            features_extractor_class=extractor_type,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = PPO(
            "MultiInputPolicy", 
            train_env,
            verbose=1,
            learning_rate=linear_schedule(float(config.parameters.learning_rate)),
            gamma=config.parameters.gamma,
            n_steps=config.parameters.n_steps,
            batch_size=config.parameters.batch_size,
            ent_coef=config.parameters.ent_coef,
            clip_range=config.parameters.clip_range,
            gae_lambda=config.parameters.gae_lambda,
            vf_coef=config.parameters.vf_coef,
            n_epochs=config.parameters.n_epochs,
            max_grad_norm=config.parameters.max_grad_norm,
            policy_kwargs=policy_kwargs,
            target_kl=config.parameters.target_kl,
            tensorboard_log=os.path.join(self.folder_path, config.settings.tensorboard_log),
            device=config.settings.device
        )

        model.learn(total_timesteps=config.parameters.total_timesteps, callback=callback)
        
        save_path = os.path.join(self.folder_path, config.settings.model_save_path)
        stats_path = os.path.join(self.folder_path, "vec_normalize.pkl")
        
        model.save(save_path)
        
        if hasattr(self.train_env, 'save'):
            self.train_env.save(stats_path)
            print(f"Stats saved to {stats_path}")

        return model