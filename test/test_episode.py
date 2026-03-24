import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manager.episode import Episode, StepData
from finance.ohlcv import OHLCV

@pytest.fixture
def sample_dataframe():
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.linspace(100, 110, 10),
        'high': np.linspace(105, 115, 10),
        'low': np.linspace(95, 105, 10),
        'close': np.linspace(102, 112, 10),
        'volume': np.linspace(1000, 2000, 10),
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })

def test_episode_initialization_invalid_window(sample_dataframe):
    with pytest.raises(ValueError, match="Window size is greater than data length"):
        Episode(data=sample_dataframe, window_size=20)

def test_episode_initialization(sample_dataframe):
    episode = Episode(data=sample_dataframe, window_size=3)
    
    assert episode.window_size == 3
    assert episode.current_index == 3
    assert not episode.out_of_data
    # Exclude basic OHLCV cols
    assert set(episode._feature_cols) == {'feature1', 'feature2'}

def test_episode_next(sample_dataframe):
    window_size = 3
    episode = Episode(data=sample_dataframe, window_size=window_size)
    
    # Check first step (index 3)
    step1 = episode.next()
    
    assert isinstance(step1, StepData)
    assert isinstance(step1.ohlcv, OHLCV)
    assert step1.ohlcv.open == sample_dataframe.iloc[3]['open']
    assert not step1.no_more_data
    assert step1.observation.shape == (window_size, 2) # 3 rows, 2 features
    
    # Check observation sliced correctly (indices 0, 1, 2)
    expected_obs = sample_dataframe.iloc[0:3][['feature1', 'feature2']].values
    np.testing.assert_array_almost_equal(step1.observation, expected_obs)

def test_episode_exhaustion(sample_dataframe):
    window_size = 8
    episode = Episode(data=sample_dataframe, window_size=window_size)
    
    # Step at index 8
    step1 = episode.next()
    assert not step1.no_more_data
    
    # Step at index 9 (last element)
    step2 = episode.next()
    assert step2.no_more_data
    assert episode.out_of_data
    
    # Step beyond exhaustion
    with pytest.raises(ValueError, match="Out of data"):
        episode.next()
