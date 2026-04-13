import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.market import Market

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
    
    test_venv = DummyVecEnv([_init])
    
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

def signal_strategy_buy_and_hold(env):
    obs = env.reset()
    done = False
    history = []
    action = 1
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        current_info = info
        history.append(
            {
                "timestamp": current_info["timestamp"],
                "equity": current_info["equity"],
            }
        )
        done = terminated or truncated
    df = pd.DataFrame(history)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    df = df.resample('1h').ffill()
    return df


def signal_model(model, env):
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
                "action": current_info["action"],
                "price": current_info["price"]
            }
        )
    df = pd.DataFrame(history)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    df = df.resample('1h').ffill()
    return df
