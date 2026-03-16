from binance.client import Client
import pandas as pd
import os

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
    
    os.makedirs('../../data/raw', exist_ok=True)
    csv_path = f'../../data/raw/btc_{tf}_raw.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ {len(df):,} rows → {csv_path}")

print("🎉 Complete multi-timeframe dataset ready!")
