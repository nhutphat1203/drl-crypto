import os
import pandas as pd
import numpy as np
import ccxt
import time

def download_data_by_range(exchange_id, symbol, start_year, end_year, timeframe='15m'):
    # 1. Khởi tạo sàn
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
    })

    # 2. Thiết lập mốc thời gian bắt đầu và kết thúc
    since_str = f"{start_year}-01-01T00:00:00Z"
    until_str = f"{end_year}-12-31T23:59:59Z"
    
    since = exchange.parse8601(since_str)
    until = exchange.parse8601(until_str)
    
    print(f"[*] Đang tải {symbol} từ {exchange_id}")
    print(f"[*] Khoảng thời gian: {start_year} -> {end_year}")
    
    all_ohlcv = []
    
    while since < until:
        try:
            # Tải dữ liệu
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not ohlcv or len(ohlcv) == 0:
                break
            
            # Kiểm tra xem nến cuối cùng có vượt quá mốc end_year không
            last_candle_time = ohlcv[-1][0]
            if last_candle_time > until:
                # Chỉ lấy những nến nằm trong khoảng yêu cầu
                ohlcv = [candle for candle in ohlcv if candle[0] <= until]
                all_ohlcv.extend(ohlcv)
                break
            
            all_ohlcv.extend(ohlcv)
            since = last_candle_time + 1
            
            # Log tiến độ mỗi lần tải (batch)
            current_dt = exchange.iso8601(last_candle_time)
            print(f" -> Đã lấy đến: {current_dt} | Tổng cộng: {len(all_ohlcv)} nến")
            
            # Nghỉ để tránh lỗi 429 (Too many requests)
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"[!] Lỗi phát sinh: {e}")
            time.sleep(5) # Nghỉ lâu hơn nếu có lỗi mạng
            continue

    # 3. Xử lý với Pandas
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    
    return df

def download_and_save_crypto_data(symbol, start_year, end_year, timeframe='1h', exchange_id='binance', folder='data'):
    """
    Hàm tải dữ liệu crypto và lưu vào file CSV.
    Args:
        symbol (str): Cặp giao dịch, ví dụ 'BTC/USDT', 'ETH/USDT'.
        start_year (int): Năm bắt đầu.
        end_year (int): Năm kết thúc.
        timeframe (str): Khung thời gian, mặc định '1h'.
        exchange_id (str): Tên sàn, mặc định 'binance'.
        folder (str): Thư mục lưu file, mặc định 'data'.
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu vừa tải.
    """
    
    print(f"⏳ Đang tải {symbol} từ {start_year} đến {end_year} ({timeframe})...")
    
    # 1. Tải dữ liệu
    df = download_data_by_range(
        exchange_id=exchange_id, 
        symbol=symbol, 
        start_year=start_year, 
        end_year=end_year, 
        timeframe=timeframe
    )
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Đã tạo thư mục: {folder}")
    
    safe_symbol = symbol.replace('/', '_')
    file_name = f"{exchange_id}_{safe_symbol}_{start_year}_{end_year}_{timeframe}.csv"
    file_path = os.path.join(folder, file_name)
    
    df.to_csv(file_path)
    print(f"✅ Đã lưu thành công: {file_path}")
    print(f"📊 Kích thước dữ liệu: {df.shape}")
    
    return df

download_and_save_crypto_data(
    symbol='BTC/USDT', 
    start_year=2020, 
    end_year=2026,
    timeframe='15m'
)
download_and_save_crypto_data(
    symbol='ETH/USDT', 
    start_year=2020, 
    end_year=2026,
    timeframe='15m'
)