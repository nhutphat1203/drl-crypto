import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from finance.account import Account, AccountState
from finance.ohlcv import create_ohlcv

@pytest.fixture
def base_account():
    # Account no longer takes explicit seed on initialization
    return Account(initial_balance=10000.0)

@pytest.fixture
def sample_ohlcv():
    timestamp = pd.Timestamp(datetime(2023, 1, 1))
    return create_ohlcv(timestamp=timestamp, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0)

def test_account_initialization(base_account):
    assert base_account.initial_balance == 10000.0
    assert base_account.balance == 10000.0
    assert base_account.equity == 10000.0
    assert base_account.prev_equity == 10000.0
    assert base_account.crypto_quantity == 0.0
    assert base_account.fee_open_total == 0.0
    assert base_account.fee_close_total == 0.0
    assert base_account.total_trades == 0
    assert isinstance(base_account.np_random, np.random.Generator)

def test_account_reset(base_account):
    base_account.balance = 5000.0
    base_account.equity = 6000.0
    base_account.crypto_quantity = 10.0
    base_account.total_trades = 5
    
    base_account.reset(seed=42)
    
    assert base_account.balance == 10000.0
    assert base_account.equity == 10000.0
    assert base_account.crypto_quantity == 0.0
    assert base_account.total_trades == 0
    # ensure seed actually worked via random check (optional but safe)

def test_step_buy(base_account, sample_ohlcv):
    action = 0.5  # Buy with 50% of balance
    initial_balance = base_account.balance
    
    state: AccountState = base_account.step(action, sample_ohlcv)
    
    assert base_account.total_trades == 1
    assert base_account.balance < initial_balance  # Balance should decrease
    assert base_account.crypto_quantity > 0.0      # Quantity should be positive
    assert base_account.fee_open_total > 0.0       # Open fee should be charged
    
    expected_spend = initial_balance * action
    assert expected_spend == 5000.0
    assert base_account.balance == initial_balance - expected_spend
    
    assert state.reward == base_account.reward()
    assert not state.terminated

def test_step_sell(base_account, sample_ohlcv):
    # First buy to get some crypto
    base_account.step(1.0, sample_ohlcv)
    initial_crypto = base_account.crypto_quantity
    assert initial_crypto > 0.0
    
    # Then sell
    action = -0.5  # Sell 50% of crypto
    state: AccountState = base_account.step(action, sample_ohlcv)
    
    assert base_account.total_trades == 2
    assert base_account.crypto_quantity < initial_crypto # Quantity should decrease
    assert base_account.fee_close_total > 0.0            # Close fee should be charged
    assert not state.terminated
    
def test_step_hold(base_account, sample_ohlcv):
    initial_balance = base_account.balance
    
    state = base_account.step(0.0, sample_ohlcv)
    
    assert base_account.total_trades == 0
    assert base_account.balance == initial_balance
    assert base_account.crypto_quantity == 0.0
    assert state.reward == 0.0

def test_get_stats(base_account, sample_ohlcv):
    base_account.step(0.5, sample_ohlcv)
    stats = base_account.get_stats()
    
    assert isinstance(stats, dict)
    assert 'balance' in stats
    assert 'crypto_quantity' in stats
    assert 'equity' in stats
    assert 'fee_open_total' in stats
    assert 'fee_close_total' in stats
    assert 'total_trades' in stats
    
    assert stats['total_trades'] == 1
    assert stats['balance'] == base_account.balance
    
def test_account_termination(base_account, sample_ohlcv):
    # Setup bad trade leading to significant loss to trigger terminated logic
    base_account.balance = 500.0  # Bring balance down
    base_account.initial_balance = 10000.0
    base_account.equity = 900.0 # equity <= initial * 0.1
    # Trigger logic check
    state = base_account.step(0.0, sample_ohlcv)
    assert state.terminated is True
