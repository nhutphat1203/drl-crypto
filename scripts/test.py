import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backtest import backtest
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DRL cho giao dịch tiền mã hóa")
    
    parser.add_argument('--folder_signal', type=str, required=True, 
                        help='Đường dẫn thư mục lưu signal (ví dụ: models/gru_tick_720/signal)')
    return parser.parse_args()

def load_data(path):
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)

def get_return(df: pd.DataFrame):
    data = df["equity"].resample("1h").ffill()
    data = data.pct_change().fillna(0)
    return data

def test(name: str, folder_path, returns, benchmark):
    print(f'Backtesting {name} model')
    stats, fig = backtest.evaluate_model_vs_benchmark(
       model_returns=returns, 
        benchmark_returns=benchmark, 
    )
    fig.write_image(f"{os.path.join(folder_path, name)}.png", scale=3)
    csv_file = os.path.join(folder_path, f"{name}_stats.csv")
    stats.to_csv(csv_file)
    print(f"Saved stats CSV to: {csv_file}")
    print(stats)
    
if __name__ == "__main__":
    args = parse_args()
    path = args.folder_signal
    bh_path = os.path.join(path, "bh_signal.csv")
    best_model_path = os.path.join(path, "best_signal.csv")
    normal_model_path = os.path.join(path, "normal_signal.csv")
    bh_data = load_data(path=bh_path)
    best_model_data = load_data(path=best_model_path)
    normal_model_data = load_data(path=normal_model_path)
    
    bh_returns = get_return(bh_data)
    b_returns = get_return(best_model_data)
    n_returns = get_return(normal_model_data)
    
    test(name="normal", folder_path=path, returns=n_returns, benchmark=bh_returns)
    test(name="best", folder_path=path, returns=n_returns, benchmark=bh_returns)