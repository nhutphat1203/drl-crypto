from dataclasses import dataclass
import pandas as pd

@dataclass
class OHLCV:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    
def create_ohlcv(timestamp: pd.Timestamp, open: float, high: float, low: float, close: float, volume: float) -> OHLCV:
    """Create an OHLCV object from the given parameters.
    
    Args:
        timestamp (pd.Timestamp): The timestamp of the OHLCV data.
        open (float): The opening price of the asset.
        high (float): The highest price of the asset during the period.
        low (float): The lowest price of the asset during the period.
        close (float): The closing price of the asset.
        volume (float): The trading volume of the asset during the period.
    
    Returns:
        OHLCV: An instance of the OHLCV dataclass containing the provided data.
    """
    return OHLCV(timestamp=timestamp, open=open, high=high, low=low, close=close, volume=volume)