import sys
import os
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import load_config
from preprocess.dataprocess import load_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from trainer.factory import DataMetadata

if __name__ == "__main__":
    config = load_config(path='config.yaml')

    info = {
        "btc": "dataprocessed/binance_BTC_USDT_processed.csv",
    }

    datas: list[DataMetadata] = []
    print('Loading data...')
    for key, value in info.items():
        data = load_data(value)
        train_data = data[(data.index >= '2020-01-01') & (data.index < '2024-01-01')]
        eval_data = data[(data.index >= '2024-01-01') & (data.index < '2024-06-01')]
        test_data = data[(data.index >= '2024-06-01') & (data.index <= '2026-03-31')]
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