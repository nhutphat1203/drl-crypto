import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backtest.backtest import create_env_for_model, strategy_buy_and_hold, backtest
from stable_baselines3.common.vec_env import DummyVecEnv

@pytest.fixture
def dummy_data():
    dates = pd.date_range('2023-01-01', periods=50, freq='15min')
    df = pd.DataFrame({
        'open': np.linspace(100, 150, 50),
        'high': np.linspace(105, 155, 50),
        'low': np.linspace(95, 145, 50),
        'close': np.linspace(102, 152, 50),
        'volume': np.random.rand(50) * 1000
    }, index=dates)
    return df

@patch('backtest.backtest.os.path.exists')
@patch('backtest.backtest.VecNormalize.load')
def test_create_env_for_model(mock_vec_load, mock_exists, dummy_data):
    # Setup mock validation logic
    mock_exists.return_value = True
    
    mock_env_instance = MagicMock()
    mock_vec_load.return_value = mock_env_instance
    
    # Function execution
    res_env = create_env_for_model(
        data=dummy_data,
        initial_balance=1000.0,
        window_size=5,
        episode_length=50,
        vec_normalize_path="mock/path.pkl"
    )
    
    # Asserts
    mock_exists.assert_called_once_with("mock/path.pkl")
    mock_vec_load.assert_called_once()
    
    assert res_env == mock_env_instance
    assert res_env.training == False
    assert res_env.norm_reward == False
    assert res_env.clip_reward == 10000.0

@patch('pandas.DataFrame.to_csv')
def test_strategy_buy_and_hold(mock_to_csv, dummy_data):
    # Run historical sequence
    res_df = strategy_buy_and_hold(
        data=dummy_data,
        initial_balance=1000.0,
        window_size=5,
        episode_length=len(dummy_data)
    )
    
    assert isinstance(res_df, pd.DataFrame)
    assert not res_df.empty
    
    # Validate feature inclusions mapping strategy calculations correctly
    assert 'equity' in res_df.columns
    assert 'price' in res_df.columns
    assert 'action' in res_df.columns
    assert 'reward' in res_df.columns
    
    # 1.0 (Buy) was invoked on the first step, then 0.0 (Hold)
    assert res_df.iloc[0]['action'] == 1.0
    
    # Ensure standard resampling limits gaps (test frequency fills accurately)
    mock_to_csv.assert_called_once_with("buy_and_hold.csv")

@patch('quantstats.reports.html')
@patch('os.makedirs')
def test_backtest_report_generator(mock_makedirs, mock_qs_html, dummy_data):
    # Establish a simulated valid execution from `strategy_buy_and_hold` generating `equity` tracking log return boundaries natively
    test_strategy_df = pd.DataFrame({
        'equity': np.linspace(1000, 1500, 50)
    }, index=dummy_data.index)
    
    backtest(test_strategy_df, "Mock_Strategy", "mock_file")
    
    # Assert
    mock_makedirs.assert_called_once_with('reports', exist_ok=True)
    mock_qs_html.assert_called_once()
    
    kwargs = mock_qs_html.call_args[1]
    assert kwargs['title'] == 'Backtest'
    assert kwargs['output'] == 'reports/mock_file_back_test_report.html'
