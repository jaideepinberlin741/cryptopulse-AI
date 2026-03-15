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

# csv_path = '../../data/raw/btc_1h_lib.csv'

# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the project's root directory (by going up two levels)
project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))

# Now, build the full, correct path to the target file
csv_path = os.path.join(project_root, 'data', 'raw', 'btc_1h_lib.csv')

# This part is still good practice to ensure the directory exists!
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(klines)

print(f"Saved to {csv_path}")
