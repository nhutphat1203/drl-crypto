import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer.trainer import Trainer, linear_schedule
from config import Config
from stable_baselines3.common.callbacks import CallbackList

def test_linear_schedule():
    scheduler = linear_schedule(initial_value=0.01)
    
    assert scheduler(1.0) == 0.01   # Start of training
    assert scheduler(0.5) == 0.005  # Mid training
    assert scheduler(0.0) == 0.0    # End of training

@pytest.fixture
def dummy_config():
    cfg = {
        "settings": {
            "folder_path": "/tmp",
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

@patch('trainer.trainer.StopTrainingOnNoModelImprovement')
@patch('trainer.trainer.EvalCallback')
@patch('trainer.trainer.CheckpointCallback')
@patch('trainer.trainer.PPO')
def test_trainer_train_mocked(mock_ppo_class, mock_checkpoint, mock_eval, mock_stop, dummy_config):
    # Create trainer
    mock_train_env = MagicMock()
    mock_val_env = MagicMock()
    
    # We add save attr to mimic VecNormalize for train_env
    mock_train_env.save = MagicMock()
    
    trainer_instance = Trainer(
        train_env=mock_train_env,
        val_env=mock_val_env,
        config=dummy_config
    )
    
    # Prepare Mocked PPO Object to be returned
    mock_model_instance = MagicMock()
    mock_ppo_class.return_value = mock_model_instance
    
    # Act
    returned_model = trainer_instance.train()
    
    # Assert
    # 1. Check PPO was initialized with correct arguments
    mock_ppo_class.assert_called_once()
    args, kwargs = mock_ppo_class.call_args
    assert args[0] == "MultiInputPolicy"
    assert args[1] == mock_train_env
    assert kwargs['gamma'] == 0.99
    assert kwargs['device'] == 'cpu'
    
    # 2. Check model.learn is called correctly
    mock_model_instance.learn.assert_called_once()
    learn_kwargs = mock_model_instance.learn.call_args[1]
    assert learn_kwargs['total_timesteps'] == 1000
    assert isinstance(learn_kwargs['callback'], CallbackList)
    
    # 3. Check model.save and env.save called
    mock_model_instance.save.assert_called_once()
    save_path = mock_model_instance.save.call_args[0][0]
    assert save_path.endswith('models/ppo')
    mock_train_env.save.assert_called_once()

