from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from finance.ohlcv import OHLCV, create_ohlcv

@dataclass
class StepData:
    ohlcv: OHLCV
    observation: np.ndarray
    no_more_data: bool

@dataclass
class Episode:
    data: pd.DataFrame
    window_size: int
    current_index: int = field(init=False)
    out_of_data: bool = field(init=False, default=False)
    _feature_cols: list[str] = field(init=False)

    def __post_init__(self):
        self.current_index = self.window_size
        
        if self.window_size > len(self.data):
            raise ValueError("Window size is greater than data length")
            
        cols_to_exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        self._feature_cols = [col for col in self.data.columns if col not in cols_to_exclude]

    def next(self):
        if self.out_of_data:
            raise ValueError("Out of data")
            
        current_tick_data = self.data.iloc[self.current_index]  
        
        ohlcv = create_ohlcv(
            open=current_tick_data['open'],
            high=current_tick_data['high'],
            low=current_tick_data['low'],
            close=current_tick_data['close'],
            volume=current_tick_data['volume'],
            timestamp=current_tick_data['timestamp']
        )
        
        observation = self.data.iloc[self.current_index - self.window_size : self.current_index][self._feature_cols].values
        
        self.current_index += 1
        self.out_of_data = self.current_index >= len(self.data)
        
        return StepData(ohlcv=ohlcv, observation=observation, no_more_data=self.out_of_data)