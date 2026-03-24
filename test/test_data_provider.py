import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manager.data_provider import DataProvider
from data_manager.episode import Episode

@pytest.fixture
def long_dataframe():
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.ones(50),
        'high': np.ones(50),
        'low': np.ones(50),
        'close': np.ones(50),
        'volume': np.ones(50),
        'feature1': np.random.rand(50)
    })

def test_dataprovider_invalid_params(long_dataframe):
    # window_size > data length
    with pytest.raises(ValueError):
        DataProvider(data=long_dataframe, window_size=60, tick_for_episode=10)
    
    # window_size > tick_for_episode
    with pytest.raises(ValueError):
        DataProvider(data=long_dataframe, window_size=15, tick_for_episode=10)
        
    # tick_for_episode > data length
    with pytest.raises(ValueError):
        DataProvider(data=long_dataframe, window_size=5, tick_for_episode=60)

def test_dataprovider_initialization(long_dataframe):
    provider = DataProvider(data=long_dataframe, window_size=5, tick_for_episode=20)
    # List indices should range from (20 - 1) to 49, so length is 50 - 19 = 31
    assert len(provider.list_indices) == 31
    # Check if list is bounded properly
    assert all(19 <= i < 50 for i in provider.list_indices)

def test_dataprovider_full_episode(long_dataframe):
    provider = DataProvider(data=long_dataframe, window_size=5, tick_for_episode=10, use_full_for_one_episode=True)
    
    # Because use_full_for_one_episode is True, tick_for_episode is overridden by data len
    assert provider.tick_for_episode == 50
    assert len(provider.list_indices) == 1
    assert provider.list_indices[0] == 49

def test_dataprovider_next_episode(long_dataframe):
    provider = DataProvider(data=long_dataframe, window_size=10, tick_for_episode=25)
    
    episode1 = provider.next_episode()
    assert isinstance(episode1, Episode)
    assert len(episode1.data) == 25
    assert episode1.window_size == 10
    
    assert provider.current_index == 1
    
def test_dataprovider_reshuffle(long_dataframe):
    # Setup highly restricted provider
    provider = DataProvider(data=long_dataframe, window_size=10, tick_for_episode=49)
    # len(list_indices) = 50 - 48 = 2
    
    # Explicitly seed to ensure reproducible shuffling logic if necessary
    provider.reset(seed=42)
    
    # Store old sequence to detect shuffle (though pseudo-random)
    old_indices = provider.list_indices.copy()
    
    ep1 = provider.next_episode() # Uses index 0
    ep2 = provider.next_episode() # Uses index 1
    assert provider.current_index == 2
    
    ep3 = provider.next_episode() # Should reshuffle since it hits 2
    assert provider.current_index == 1
