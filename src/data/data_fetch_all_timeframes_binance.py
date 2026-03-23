from binance.client import Client
import pandas as pd
import os

script_path = os.path.abspath(__file__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

output_dir = os.path.join(project_root, 'data', 'raw')

client = Client()
timeframes = ['15m', '1h', '4h', '1d'] 

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
    print(f" {len(df):,} rows → {csv_path}")

print(" Complete multi-timeframe dataset ready!")