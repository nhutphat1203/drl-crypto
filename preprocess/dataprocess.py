import pandas as pd
import numpy as np
import talib as ta

def load_data(path):
    return pd.read_csv(path, index_col="datetime", parse_dates=True)

def pre_process(df):
    data = df.copy()
    EPSILON = 1e-8
    
    delta = data.index.to_series().diff()
    delta_time = delta.dt.total_seconds() / 3600
    delta_time = delta_time.fillna(1).astype(int)
    data['log_time_gap'] = np.log(delta_time)

    # 1. Temporal Features
    hours = data.index.hour
    day_of_week = data.index.dayofweek
    data['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    data['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    data['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    data['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # 1. Price structure
    high_low_range = data['high'] - data['low'] + EPSILON
    data['body_ratio'] = (data['close'] - data['open']) / high_low_range
    max_open_close = data[['open', 'close']].max(axis=1)
    data['upper_shadow_ratio'] = (data['high'] - max_open_close) / high_low_range

    # 2. Multi-horizon Momentum
    horizons = [1, 4, 12, 24, 168]
    for h in horizons:
        data[f'log_ret_{h}'] = np.log(data['close'] / data['close'].shift(h))

    # 3. Volatility & Risk Representation
    data['volatility_4'] = data['log_ret_1'].rolling(window=4).std()
    
    volatility_horizons = [24, 168, 720]
    for h in volatility_horizons:
        volatility_h = data['log_ret_1'].rolling(window=h).std()
        data[f'volatility_{h}_ratio'] = data['volatility_4'] / (volatility_h + EPSILON)

    # Normalized Spread
    data['spread_hl_norm'] = np.log(data['high'] / (data['low'] + EPSILON)) / (data['volatility_4'] + EPSILON)

    # 4. Microstructure & Volume Dynamics
    typical_price = ta.TYPPRICE(data['high'], data['low'], data['close'])
    tp_vol = typical_price * data['volume']
    
    rolling_tp_vol = ta.SUM(tp_vol, timeperiod=24)
    rolling_vol = ta.SUM(data['volume'], timeperiod=24)
    
    vwap_24 = rolling_tp_vol / (rolling_vol + EPSILON)
    data['dist_vwap_24'] = (data['close'] - vwap_24) / (vwap_24 + EPSILON)

    # 5. Signed Volume Pressure
    direction = np.sign(data['log_ret_1'].fillna(0))
    log_signed_volume = np.log(1 + data['volume']) * direction

    window_z = 24
    mean_vol = ta.SMA(log_signed_volume, timeperiod=window_z)
    std_vol = ta.STDDEV(log_signed_volume, timeperiod=window_z, nbdev=1)

    data['signed_vol_pressure'] = (log_signed_volume - mean_vol) / (std_vol + EPSILON)

    data = data.loc[(data.index >= '2020-01-01') & (data.index <= '2026-03-31')]
    data = data.dropna()
    return data

def preprocess_and_save(path_data, save_path):
    print(f"Loading data from {path_data}")
    data = load_data(path_data)
    print(f"Preprocessing data from {path_data}")
    data = pre_process(data)
    print(f"Saving data to {save_path}")
    data.to_csv(save_path)
    print(f"Data saved to {save_path}")
    return data