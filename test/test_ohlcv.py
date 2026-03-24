from datetime import datetime
import pandas as pd
import sys
import os

# Add finance to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'finance'))
from ohlcv import OHLCV, create_ohlcv

def test_initialization():
    timestamp = pd.Timestamp(datetime(2023, 1, 1))
    ohlcv = OHLCV(timestamp=timestamp, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0)
    
    assert ohlcv.timestamp == timestamp
    assert ohlcv.open == 100.0
    assert ohlcv.high == 105.0
    assert ohlcv.low == 95.0
    assert ohlcv.close == 102.0
    assert ohlcv.volume == 1000.0

def test_create_ohlcv():
    timestamp = pd.Timestamp(datetime(2023, 1, 2))
    ohlcv = create_ohlcv(timestamp=timestamp, open=200.0, high=210.0, low=190.0, close=205.0, volume=5000.0)
    
    assert isinstance(ohlcv, OHLCV)
    assert ohlcv.timestamp == timestamp
    assert ohlcv.open == 200.0
    assert ohlcv.high == 210.0
    assert ohlcv.low == 190.0
    assert ohlcv.close == 205.0
    assert ohlcv.volume == 5000.0
