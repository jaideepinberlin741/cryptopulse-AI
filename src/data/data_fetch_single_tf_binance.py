from binance.client import Client
import csv
import os

# Ensure data/raw exists
os.makedirs('data/raw', exist_ok=True)

client = Client()  # No key needed
klines = client.get_historical_klines(
    'BTCUSDT', 
    Client.KLINE_INTERVAL_1HOUR, 
    "1 Jan, 2020"
)

columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']

csv_path = '../../data/raw/btc_1h_lib.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(klines)

print(f"Saved to {csv_path}")
