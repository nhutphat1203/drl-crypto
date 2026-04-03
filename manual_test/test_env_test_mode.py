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
    
    data_path = "dataprocessed/binance_BTC_USDT_processed.csv"
    try:
        df = pd.read_csv(data_path, index_col="datetime", parse_dates=True)
        # Giới hạn data lại còn 1500 dòng để test nhanh chạy xuyên nhiều episodes 
        # (trong test_mode, cả dataset là 1 episode)
        df_limited = df.head(1500)
        print(f"Đã tải {len(df_limited)} dòng dữ liệu từ {data_path} (đã giới hạn).")
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return

    print("\n=== KHỞI TẠO MÔI TRƯỜNG TEST MODE ===")
    env = Market(
        df=df_limited,
        initial_balance=config.model_env.initial_balance,
        window_size=config.model_env.window_size,
        episode_length=config.model_env.tick_per_episode, # Sẽ bị override bởi DataProvider.use_full_for_one_episode = True
        test_mode=True,  # BẬT TEST MODE
        name="BTC_USDT"
    )
    
    seed = config.model_env.seed
    obs, info = env.reset(seed=seed)
    
    print(f"Test_Mode = True (One full sequence per episode)")
    print(f"Trực tiếp tạo Episode dài: {env.episode_length} ticks (bằng data length = {len(df_limited)})")
    
    print("\n=== BẮT ĐẦU CHẠY THỬ NGHIỆM ĐÁNH GIÁ TRÊN TOÀN BỘ DATA ===")
    
    episode_count = 1
    total_steps_executed = 0
    # Chạy mô phỏng vòng qua vài episodes để verify logic chạy lại
    target_episodes = 2 
    
    while episode_count <= target_episodes:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_steps_executed += 1
        
        # Chỉ print đầu kỳ và những steps sát vùng kết thúc
        if total_steps_executed % 300 == 1 or terminated or truncated:
            print(f"[Step {total_steps_executed:04d} - Ep {episode_count}] "
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
            print(f">>> Môi trường đã kết thúc Test Episode {episode_count} ở step thứ {total_steps_executed}. Đang Reset...")
            obs, info = env.reset()
            episode_count += 1
            total_steps_executed = 0 # reset nội bộ theo episode
            
    print("\n=== HOÀN TẤT QUA TỚI TEST ===")
    print("Môi trường test_mode=True đã hoàn tất thành công việc load full dataset vào một episode duy nhất.")

if __name__ == "__main__":
    main()
