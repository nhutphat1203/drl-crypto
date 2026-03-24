import pytest
import pandas as pd
import numpy as np
import sys
import os
import gymnasium as gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.market import Market

@pytest.fixture
def dummy_market_data():
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(105, 115, 100),
        'low': np.linspace(95, 105, 100),
        'close': np.linspace(102, 112, 100),
        'volume': np.random.rand(100) * 1000,
        'custom_feat': np.random.rand(100)
    })

@pytest.fixture
def env(dummy_market_data):
    # 1 feature custom_feat
    return Market(
        df=dummy_market_data,
        initial_balance=10000.0,
        window_size=10,
        episode_length=30,
        test_mode=False
    )

def test_market_initialization(env):
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0

    assert isinstance(env.observation_space, gym.spaces.Dict)
    # The dataframe has exactly 1 custom feature after exclusion
    assert env.observation_space['time_series'].shape == (10, 1)
    assert env.observation_space['portfolio_features'].shape == (4,)

def test_market_reset(env):
    obs, info = env.reset(seed=42)
    
    assert isinstance(obs, dict)
    assert 'time_series' in obs
    assert 'portfolio_features' in obs
    assert obs['time_series'].shape == (10, 1)
    assert obs['portfolio_features'].shape == (4,)
    
    assert isinstance(info, dict)
    assert 'balance' in info
    assert info['balance'] == 10000.0

def test_market_step(env):
    env.reset(seed=42)
    
    # Perform an action
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    
    assert not terminated
    assert info['total_trades'] == 1

def test_market_truncation(dummy_market_data):
    # Setting an episode_length exactly equal to the window size plus a small margin 
    # to test truncation when episode iterator is exhausted.
    env = Market(
        df=dummy_market_data,
        initial_balance=10000.0,
        window_size=10,
        episode_length=15, # Episode will exhaust quickly
        test_mode=False
    )
    env.reset(seed=42)
    
    # Since window_size is 10, episode index starts at 10. Data len is episode_length = 15.
    # It can only step a few times before exhaustion.
    steps = 0
    truncated = False
    
    while steps < 10 and not truncated:
        action = np.array([0.0], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    
    # It should truncate after (15 - 10) steps = 5 steps. The exact count relies on indexing.
    # Either way, verification is ensuring truncated becomes true.
    assert truncated is True
