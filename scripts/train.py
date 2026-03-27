from torch.backends.cudnn import benchmark
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.split_data import train_test_split, train_eval_test_split
from trainer.trainer import Trainer
from config import load_config
from preprocess.dataprocess import load_data
from trainer.factory import DataMetadata, get_trainer
from backtest.backtest_strategy import backtest_model, create_env_for_model, Benchmark, strategy_buy_and_hold
from stable_baselines3 import PPO

if __name__ == "__main__":
    print('Loading config...')
    config = load_config(path="config.yaml")

    info = {
        "btc": "dataprocessed/binance_BTC_USDT_2020_2026_15m_processed.csv",
        "eth": "dataprocessed/binance_ETH_USDT_2020_2026_15m_processed.csv"
    }

    datas: list[DataMetadata] = []
    test = []
    print('Loading data...')
    for key, value in info.items():
        data = load_data(value)
        train_data, eval_data, test_data = train_eval_test_split(data, train_ratio=0.8, eval_ratio=0.1)
        datas.append(DataMetadata(train_data, eval_data))
        test.append({"name": key, "data": test_data})

    traning = False
    
    if traning:
        print('Creating trainer...')
        trainer = get_trainer(datas, config)
        print('Training...')
        trainer.train()
        print('Training completed!')

    model_path = os.path.join(config.settings.folder_path, config.settings.best_model_save_path, "best_model.zip")
    model_folder_name = "gru_advanced"
    benchmark_name = "B&H"
    for i, data in enumerate(test):
        print(f'\n[{i+1}/{len(test)}] Creating environment for {data["name"]}...')
        env = create_env_for_model(
            data["data"], 
            config.model_env.initial_balance, 
            config.model_env.window_size, 
            os.path.join(config.settings.folder_path, "vec_normalize.pkl")
        )
        benchmark = Benchmark(
            name=benchmark_name,
            returns=strategy_buy_and_hold(data["data"], config.model_env.initial_balance, config.model_env.window_size)
        )
        print(f'Loading model from {model_path}...')
        model = PPO.load(model_path, env=env) 
        print(f'Backtesting {data["name"]}...')
        backtest_folder = os.path.join("reports", model_folder_name)
        backtest_model(model=model, env=env, name_strategy="Agent", file_name=data["name"], backtest_folder=backtest_folder, benchmark=benchmark)
        env.close()



    