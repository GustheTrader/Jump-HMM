import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_ohlcv(symbol, timeframe='1m', months=6):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        exchange.load_markets()
    except Exception:
        print("Standard Binance failed/blocked. Switching to Binance US...")
        exchange = ccxt.binanceus({'enableRateLimit': True})
    
    # Calculate start time
    now = datetime.utcnow()
    start_time = now - timedelta(days=30 * months)
    since = int(start_time.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000 # Binance max limit per request
    
    print(f"Fetching {months} months of {timeframe} data for {symbol}...")
    
    while since < int(now.timestamp() * 1000):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Move 'since' to the timestamp of the last bar + 1ms
            since = ohlcv[-1][0] + 1
            
            # Log progress
            last_date = datetime.utcfromtimestamp(ohlcv[-1][0]/1000).strftime('%Y-%m-%d %H:%M')
            print(f"Progress: Fetched up to {last_date} ({len(all_ohlcv)} total rows)")
            
            # Short sleep to respect rate limit even though enableRateLimit is True
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5) # Wait and retry
            continue

    if not all_ohlcv:
        print("No data fetched.")
        return None

    # Process into DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    filename = f"{symbol.replace('/', '')}_{timeframe}_{months}mo.csv"
    df.to_csv(filename)
    print(f"Completed! Saved {len(df)} rows to {filename}")
    return df

if __name__ == "__main__":
    # Fetch BTC and ETH
    btc_df = fetch_binance_ohlcv('BTC/USDT', months=6)
    eth_df = fetch_binance_ohlcv('ETH/USDT', months=6)
