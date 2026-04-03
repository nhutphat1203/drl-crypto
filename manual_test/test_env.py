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
    test_episode_length = 150 # Mỗi episode dài 150 timestep
    total_timesteps_to_test = 500 # Chạy tổng cộng 500 timestep
    
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
    
    episode_count = 1
    for i in range(1, total_timesteps_to_test + 1):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Chỉ in ra log ở những step đầu của episode và step cuối để dễ theo dõi
        if i % test_episode_length in [1, 2, 0] or terminated or truncated:
            print(f"[Step {i:03d} - Ep {episode_count}] "
                  f"Time: {step_info['timestamp']} | "
                  f"Act: {action[0]:+5.2f} | "
                  f"Rew: {reward:+5.5f} | "
                  f"Eq: ${step_info['equity']:,.2f} | "
                  f"Balance: ${step_info['balance']:,.2f} | "
                  f"Coin: {step_info['crypto_quantity']:.6f} | "
                  f"Total buy: {step_info['total_buy']} | "
                  f"Total sell: {step_info['total_sell']} | "
                  f"Price: {step_info['price']:.2f} | "
                  f"Term: {terminated} | Trunc: {truncated}")
              
        if terminated or truncated:
            print(f">>> Môi trường đã kết thúc Episode {episode_count}. Đang Reset sang Episode tiếp theo...")
            obs, info = env.reset()
            episode_count += 1
            
    print("\n=== HOÀN TẤT QUA TỚI TEST ===")
    print("Cấu trúc Observation (State) trả về từ Reset/Step:")
    print(f"- Time_series shape:     {obs['time_series'].shape}")
    print(f"- Portfolio_features shape: {obs['portfolio_features'].shape}")
    print(f"Tổng số Episode đã trải qua: {episode_count - 1}")

if __name__ == "__main__":
    main()
