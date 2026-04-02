import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import quantstats as qs
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.market import Market
from preprocess.dataprocess import load_data
from preprocess.split_data import train_test_split, train_eval_test_split
from config import load_config
from dataclasses import dataclass

@dataclass
class Benchmark:
    name: str
    returns: pd.DataFrame

def create_env_normal(data: pd.DataFrame, initial_balance: float, window_size: int):
    market = Market(df=data, 
    name="test",
    initial_balance=initial_balance,
    window_size=window_size,
    episode_length=1,
    test_mode=True,
    verbose=1
    )
    return market

def create_env_for_model(
    data: pd.DataFrame, 
    initial_balance: float, 
    window_size: int, 
    vec_normalize_path: str  # Yêu cầu đường dẫn đến file thống kê đã lưu lúc train
):
    # 1. Hàm khởi tạo môi trường gốc
    def _init() -> Market:
        return Market(
            df=data, 
            name="test",
            initial_balance=initial_balance,
            window_size=window_size,
            episode_length=1,
            test_mode=True,
            verbose=1
        )
    
    # 2. Bọc bằng DummyVecEnv để đồng nhất chiều dữ liệu (batch_size = 1)
    test_venv = DummyVecEnv([_init])

    # 3. Load lớp VecNormalize cùng với các thông số (mean, var) đã học từ tập Train
    if os.path.exists(vec_normalize_path):
        test_venv = VecNormalize.load(vec_normalize_path, test_venv)
        test_venv.training = False      
        test_venv.norm_reward = False   
        test_venv.clip_reward = 10000.0  
        
        print(f"Đã load thành công môi trường test với VecNormalize từ: {vec_normalize_path}")
    else:
        raise FileNotFoundError(
            f"Không tìm thấy {vec_normalize_path}. "
            f"Bạn cần gọi lệnh `train_venv.save('đường_dẫn')` sau khi train xong!"
        )

    return test_venv

def strategy_buy_and_hold(data: pd.DataFrame, initial_balance: float, window_size: int):
    env = create_env_normal(data, initial_balance, window_size)
    obs = env.reset()
    done = False
    history = []
    action = 1.0 #Buy at first
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        action = 0.0
        history.append(
            {
                "timestamp": info["timestamp"],
                "equity": info["equity"],
            }
        )
    df = pd.DataFrame(history)
    df.index = pd.to_datetime(df["timestamp"])
    df = df[~df.index.duplicated(keep='last')]
    df = df.resample('15min').ffill()
    returns = df["equity"].pct_change().dropna()
    return returns

def backtest(df_strategy, name_strategy, file_name, backtest_folder, benchmark: Benchmark=None):
    returns = df_strategy["equity"].pct_change().dropna()
    trading_periods_per_year = 365 * 24 * 4
    returns.name = name_strategy
    rf = 0.052
    folder_path = backtest_folder
    os.makedirs(folder_path, exist_ok=True)
    name_file = f'{file_name}_back_test_report.html'
    if benchmark is not None:
        qs.reports.html(
            returns, 
            output=f'{folder_path}/{name_file}', 
            title='Backtest', 
            periods_per_year=trading_periods_per_year,
            rf=rf,
            benchmark=benchmark,
            benchmark_title=benchmark.name,
        )
    else:
        qs.reports.html(
            returns, 
            output=f'{folder_path}/{name_file}', 
            title='Backtest', 
            periods_per_year=trading_periods_per_year,
            rf=rf,
        )

def backtest_model(model, env, name_strategy, file_name, backtest_folder, benchmark: Benchmark=None):
    obs = env.reset()
    done = [False]
    history = []
    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_info = info[0]
        history.append(
            {
                "timestamp": current_info["timestamp"],
                "equity": current_info["equity"],
            }
        )
    df = pd.DataFrame(history)
    df.index = pd.to_datetime(df["timestamp"])
    df = df[~df.index.duplicated(keep='last')]
    df = df.resample('15min').ffill()
    backtest(df, name_strategy, file_name, backtest_folder, benchmark)

if __name__ == "__main__":
    config = load_config('config.yaml')
    info = {
        "btc": "dataprocessed/binance_BTC_USDT_2020_2026_15m_processed.csv",
        # "eth": "dataprocessed/binance_ETH_USDT_2020_2026_15m_processed.csv"
    }
    
    for key, value in info.items():
        if not os.path.exists(value):
            continue
        data = load_data(value)
        train_data, eval_data, test_data = train_eval_test_split(data, train_ratio=0.8, eval_ratio=0.1)
        
        b_h_df = strategy_buy_and_hold(
            data=test_data, 
            initial_balance=config.model_env.initial_balance, 
            window_size=config.model_env.window_size, 
            episode_length=config.model_env.tick_per_episode,
        )
        
        backtest(b_h_df, "B&H", f"{key}_b_h")