import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_ohlcv(symbol, timeframe='1m', months=6):
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        exchange.load_markets()
    except:
        print("Standard Binance failed/blocked. Switching to Binance US...")
        exchange = ccxt.binanceus({'enableRateLimit': True})
    
    # Use UTC for consistency
    now = datetime.utcnow()
    start_time = now - timedelta(days=30 * months)
    since = int(start_time.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000
    
    print(f"Fetching {months} months of {timeframe} data for {symbol}...")
    while since < int(now.timestamp() * 1000):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(all_ohlcv) % 10000 == 0:
                print(f"Progress: {len(all_ohlcv)} rows...")
            
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching: {e}")
            time.sleep(5)
            continue

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    safe_symbol = symbol.replace('/', '')
    filename = f"{safe_symbol}_{timeframe}_{months}mo.csv"
    df.to_csv(filename, index=False)
    print(f"Completed! Saved to {filename}")
    return df

if __name__ == "__main__":
    for asset in ['SOL/USDT', 'XRP/USDT']:
        fetch_binance_ohlcv(asset, months=6)
