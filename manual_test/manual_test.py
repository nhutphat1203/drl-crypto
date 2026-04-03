import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import load_config
from environment.market import Market

def main():
    print("=== TẢI CẤU HÌNH VÀ DỮ LIỆU ===")
    config = load_config("config.yaml")
    
    # Sử dụng file data đã được preprocess
    data_path = "dataprocessed/binance_BTC_USDT_processed.csv"
    try:
        df = pd.read_csv(data_path, index_col="datetime", parse_dates=True)
        print(f"Đã tải {len(df)} dòng dữ liệu từ {data_path}.")
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}. Vui lòng chạy scripts/preprocess_data.py trước.")
        return

    # Tùy chỉnh thông số để dễ quan sát việc chuyển episode
    test_episode_length = 28 # Mỗi episode dài 150 timestep
    total_timesteps_to_test = 8*3 # Chạy tổng cộng 500 timestep
    
    print("\n=== KHỞI TẠO MÔI TRƯỜNG ===")
    env = Market(
        df=df,
        initial_balance=config.model_env.initial_balance,
        window_size=config.model_env.window_size,
        episode_length=test_episode_length,
        test_mode=False,
        name="BTC_USDT"
    )
    
    seed = config.model_env.seed
    obs, info = env.reset(seed=seed)
    
    print(f"Episode length: {env.episode_length} ticks")
    print(f"Initial Balance: ${env.initial_balance:,.2f}")
    
    print(f"\n=== BẮT ĐẦU CHẠY THỬ NGHIỆM ({total_timesteps_to_test} STEPS) ===")
    print(f"Equity: ${info['equity']:,.2f}|Balance: ${info['balance']:,.2f}|Coin: {info['crypto_quantity']:.6f}|Total buy: {info['total_buy']}|Total sell: {info['total_sell']}")
    episode_count = 3
    while True:
        action = float(input("Action: "))
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"Price: {step_info['price']:.2f}|Equity: ${step_info['equity']:,.2f}|Balance: ${step_info['balance']:,.2f}|Coin: {step_info['crypto_quantity']:.6f}|Total buy: {step_info['total_buy']}|Total sell: {step_info['total_sell']}")
        if terminated or truncated:
            obs, info = env.reset()
            episode_count -= 1
            if episode_count == 0:
                break
            print(f"Episode {3 - episode_count} started")
            print(f"Equity: ${info['equity']:,.2f}|Balance: ${info['balance']:,.2f}|Coin: {info['crypto_quantity']:.6f}|Total buy: {info['total_buy']}|Total sell: {info['total_sell']}")
            
    print("\n=== HOÀN TẤT QUA TỚI TEST ===")
    print("Cấu trúc Observation (State) trả về từ Reset/Step:")
    print(f"- Time_series shape:     {obs['time_series'].shape}")
    print(f"- Portfolio_features shape: {obs['portfolio_features'].shape}")
    print(f"Tổng số Episode đã trải qua: {episode_count - 1}")

if __name__ == "__main__":
    main()
