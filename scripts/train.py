import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.split_data import train_test_split, train_eval_test_split
from trainer.trainer import Trainer
from config import load_config
from preprocess.dataprocess import load_data
from trainer.factory import DataMetadata, get_trainer

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
    
    print('Creating trainer...')
    trainer = get_trainer(datas, config)
    print('Training...')
    trainer.train()
    print('Training completed!')

    