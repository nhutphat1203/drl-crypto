import pathlib
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import load_config
from preprocess.dataprocess import load_data
from backtest.signal_strategy import *
from stable_baselines3 import PPO
import pandas as pd

if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath 
    
def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DRL cho giao dịch tiền mã hóa")
    
    parser.add_argument('--folder_model', type=str, required=True, 
                        help='Đường dẫn thư mục lưu model (ví dụ: models/gru_tick_720)')

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    print('Loading config...')
    config = load_config(path="config.yaml")
    
    signal_folder_name = "signal"
    folder_path = args.folder_model
    folder_data = "scaled_data"
    signal_folder = os.path.join(folder_path, signal_folder_name)
    os.makedirs(signal_folder, exist_ok=True)
    key = "btc"
    data_path = os.path.join(folder_data, f"{key}_test_scaled.csv")
    data = load_data(path=data_path)
    
    model_paths = {
        "normal": os.path.join(folder_path, config.settings.model_save_path),
        "best": os.path.join(folder_path, config.settings.best_model_save_path, "best_model.zip"),
    }
    print('Getting signal for buy and hold!')
    env = create_env_normal(data=data, initial_balance=config.model_env.initial_balance, window_size=config.model_env.window_size)
    bh_signal = signal_strategy_buy_and_hold(env)
    bh_signal.to_csv(os.path.join(signal_folder, "bh_signal.csv"))

    print('Getting signal for model!')
    for key, model_path in model_paths.items():
        env = create_env_for_model(
            data, 
            config.model_env.initial_balance, 
            config.model_env.window_size, 
            os.path.join(folder_path, "vec_normalize.pkl")
        )
        print(f'Loading model from {model_path}...')
        try:
            model = PPO.load(model_path, env=env, device='cpu') 
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue
        model_signal = signal_model(model, env)
        model_signal.to_csv(os.path.join(signal_folder, f"{key}_signal.csv"))
        env.close()


    