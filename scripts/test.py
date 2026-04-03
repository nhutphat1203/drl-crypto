import pathlib
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocess.split_data import train_eval_test_split
from config import load_config
from preprocess.dataprocess import load_data
from backtest.backtest_strategy import backtest_model, create_env_for_model, strategy_buy_and_hold
from stable_baselines3 import PPO

if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath 
    
def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DRL cho giao dịch tiền mã hóa")
    
    parser.add_argument('--folder_path', type=str, required=True, 
                        help='Đường dẫn thư mục lưu model (ví dụ: models/gru)')
    
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Đường dẫn tới file config.yaml')

    parser.add_argument('--tick_episode', type=int, default=None, 
                        help='Số tick episode muốn sử dụng')

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    print('Loading config...')
    config = load_config(path=args.config)
    
    folder_path = args.folder_path

    if args.tick_episode is not None:
        config.model_env.tick_per_episode = args.tick_episode
        folder_path = f"{folder_path}_tick_{args.tick_episode}"
    else:
        folder_path = f"{folder_path}_tick_{config.model_env.tick_per_episode}"

    info = {
        "btc": "dataprocessed/binance_BTC_USDT_processed.csv",
        "eth": "dataprocessed/binance_ETH_USDT_processed.csv"
    }

    test = []
    print('Loading data...')
    for key, value in info.items():
        data = load_data(value)
        _, _, test_data = train_eval_test_split(data, train_ratio=0.8, eval_ratio=0.1)
        test.append({"name": key, "data": test_data})

    model_paths = {
        "normal": os.path.join(folder_path, config.settings.model_save_path),
        "best": os.path.join(folder_path, config.settings.best_model_save_path, "best_model.zip"),
    }

    benchmark_name = "B&H"
    benchmarks_returns = []
    for i, data in enumerate(test):
        returns = strategy_buy_and_hold(data["data"], config.model_env.initial_balance, config.model_env.window_size)
        returns.name = benchmark_name
        benchmarks_returns.append(returns)

    for key, model_path in model_paths.items():
        for i, data in enumerate(test):
            print(f'\n[{i+1}/{len(test)}] Creating environment for {data["name"]}...')
            env = create_env_for_model(
                data["data"], 
                config.model_env.initial_balance, 
                config.model_env.window_size, 
                os.path.join(folder_path, "vec_normalize.pkl")
            )
            benchmark = benchmarks_returns[i]
            
            print(f'Loading model from {model_path}...')
            try:
                model = PPO.load(model_path, env=env, device='cpu') 
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                continue
            print(f'Backtesting {data["name"]}...')
            backtest_folder = os.path.join(folder_path, "reports", key)
            backtest_model(model=model, env=env, name_strategy="Agent", file_name=data["name"], backtest_folder=backtest_folder, benchmark=benchmark)
            env.close()


    