import argparse
import sys
import os

import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocess.split_data import train_test_split, train_eval_test_split
from trainer.trainer import Trainer
from config import load_config
from preprocess.dataprocess import load_data
from trainer.factory import DataMetadata, get_trainer
import pandas as pd
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DRL cho giao dịch tiền mã hóa")
    
    parser.add_argument('--folder_path', type=str, required=True, 
                        help='Đường dẫn thư mục lưu model (ví dụ: models/gru)')
    
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Đường dẫn tới file config.yaml')
    
    parser.add_argument('--extractor_type', type=str, choices=['CNN', 'LSTM', 'GRU'], required=True, 
                        help='Loại extractor muốn sử dụng')

    parser.add_argument('--tick_episode', type=int, default=None, 
                        help='Số tick episode muốn sử dụng')

    return parser.parse_args()

if __name__ == "__main__":
    os.makedirs("scaled_data", exist_ok=True)
    args = parse_args()

    print(f'Loading config from {args.config}...')
    config = load_config(path=args.config)

    info = {
        "btc": "dataprocessed/binance_BTC_USDT_processed.csv",
    }

    datas: list[DataMetadata] = []
    print('Loading data...')
    for key, value in info.items():
        data = load_data(value)
        train_data, eval_data, test_data = train_eval_test_split(data, train_ratio=0.8, eval_ratio=0.1)
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in train_data.columns if col not in price_cols]
        scaler = StandardScaler()
        train_data_scaled = train_data.copy()
        eval_data_scaled = eval_data.copy()
        test_data_scaled = test_data.copy()
        train_data_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])
        eval_data_scaled[feature_cols] = scaler.transform(eval_data[feature_cols])
        test_data_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
        train_data_scaled.to_csv(f"scaled_data/{key}_train_scaled.csv")
        eval_data_scaled.to_csv(f"scaled_data/{key}_eval_scaled.csv")
        test_data_scaled.to_csv(f"scaled_data/{key}_test_scaled.csv")
        datas.append(DataMetadata(train_data_scaled, eval_data_scaled))
        scaler_filename = f"scaled_data/{key}_scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        
        print(f"Đã xử lý xong data và lưu scaler cho {key} tại: {scaler_filename}")

    folder_path = args.folder_path

    if args.tick_episode is not None:
        config.model_env.tick_per_episode = args.tick_episode
        folder_path = f"{folder_path}_tick_{args.tick_episode}"
    else:
        folder_path = f"{folder_path}_tick_{config.model_env.tick_per_episode}"

    print(f'Creating trainer for {folder_path}...')
    trainer = get_trainer(datas, config, folder_path, args.extractor_type)
    
    print('Training...')
    trainer.train()
    print('Training completed!')