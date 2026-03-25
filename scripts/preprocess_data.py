import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.dataprocess import preprocess_and_save

if __name__ == "__main__":
    os.makedirs("dataprocessed", exist_ok=True)
    preprocess_and_save("data/binance_BTC_USDT_2020_2026_15m.csv", "dataprocessed/binance_BTC_USDT_2020_2026_15m_processed.csv")
    preprocess_and_save("data/binance_ETH_USDT_2020_2026_15m.csv", "dataprocessed/binance_ETH_USDT_2020_2026_15m_processed.csv")
