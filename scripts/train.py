import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocess.split_data import train_test_split, train_eval_test_split
from trainer.trainer import Trainer
from config import load_config
from preprocess.dataprocess import load_data
from trainer.factory import DataMetadata, get_trainer

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
    args = parse_args()

    print(f'Loading config from {args.config}...')
    config = load_config(path=args.config)

    info = {
        "btc": "dataprocessed/binance_BTC_USDT_processed.csv",
        "eth": "dataprocessed/binance_ETH_USDT_processed.csv"
    }

    datas: list[DataMetadata] = []
    print('Loading data...')
    for key, value in info.items():
        data = load_data(value)
        train_data, eval_data, test_data = train_eval_test_split(data, train_ratio=0.8, eval_ratio=0.1)
        datas.append(DataMetadata(train_data, eval_data))

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