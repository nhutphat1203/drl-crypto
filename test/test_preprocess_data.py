import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocess.dataprocess import load_data, pre_process, preprocess_and_save

@pytest.fixture
def dummy_market_data():
    # Need at least 673 rows due to volatility_672, so create 800 rows.
    dates = pd.date_range('2023-01-01', periods=800, freq='15min')
    
    # We add a slight drift to ensure standard deviations exist cleanly
    # (avoiding exact 0 arrays giving NaN with zero divisions).
    trend = np.linspace(100, 110, 800)
    
    # Adding a wave guarantees variance and predictable sin/cos derivations.
    wave = np.sin(np.linspace(0, 10 * np.pi, 800))
    
    return pd.DataFrame({
        'open': trend + wave,
        'high': trend + wave + 2,
        'low': trend + wave - 2,
        'close': trend + wave + 1,
        'volume': np.abs(trend + wave * 50) + 1000
    }, index=dates)

def test_pre_process(dummy_market_data):
    processed = pre_process(dummy_market_data)
    
    # Data length should have dropped due to the NA drop logic (req 672 rolling windows)
    assert len(processed) > 0
    assert len(processed) <= 800 - 672
    
    # Test important generated features presence
    expected_cols = [
        'log_time_gap', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'log_ret_1', 'log_ret_3', 'log_ret_5', 'log_ret_15', 'log_ret_60',
        'volatility_4', 'volatility_16', 'volatility_96', 'volatility_672',
        'volatility_16_ratio', 'volatility_96_ratio', 'volatility_672_ratio',
        'spread_hl_norm', 'dist_vwap_20', 'final_feature_ma'
    ]
    
    for col in expected_cols:
        assert col in processed.columns
        
    # Test NaNs were dropped
    assert not processed.isnull().values.any()
    
    # Check bounds on trig functions
    assert processed['hour_sin'].min() >= -1.0
    assert processed['hour_sin'].max() <= 1.0

@patch('preprocess.dataprocess.load_data')
@patch('pandas.DataFrame.to_csv')
def test_preprocess_and_save(mock_to_csv, mock_load_data, dummy_market_data):
    # Mock return
    mock_load_data.return_value = dummy_market_data
    
    res = preprocess_and_save("mock_path_in.csv", "mock_path_out.csv")
    
    # Verify processing successfully applied and pipeline moved forward
    assert len(res) > 0
    assert 'log_ret_1' in res.columns
    
    # Verify mocking pipeline
    mock_load_data.assert_called_once_with("mock_path_in.csv")
    mock_to_csv.assert_called_once_with("mock_path_out.csv")
