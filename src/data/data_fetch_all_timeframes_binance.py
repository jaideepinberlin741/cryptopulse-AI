from binance.client import Client
import pandas as pd
import os

# --- START: Robust Path Generation ---
# This block ensures that file paths are generated relative to the script's location,
# making the script work correctly no matter where it's run from.

# 1. Get the absolute path of the current script file
# e.g., /Users/jopanda/bootcamp-aipm/cryptopulse-AI/src/data/data_fetch_all_timeframes_binance.py
script_path = os.path.abspath(__file__)

# 2. Get the project root directory by going up the directory tree
# From /.../src/data/ -> /.../src/ -> /.../ (the project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

# 3. Construct the full, robust path to the target 'data/raw' directory
output_dir = os.path.join(project_root, 'data', 'raw')

# --- END: Robust Path Generation ---


client = Client()
timeframes = ['3m', '5m', '15m', '1h', '4h', '1d', '1w'] 

for tf in timeframes:
    print(f"Fetching BTCUSDT {tf}...")
    klines = client.get_historical_klines('BTCUSDT', tf, "1 Jan, 2020")
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Use the robust path to create the directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the robust path to create the full CSV file path
    csv_path = os.path.join(output_dir, f'btc_{tf}_raw.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"✅ {len(df):,} rows → {csv_path}")

print("🎉 Complete multi-timeframe dataset ready!")