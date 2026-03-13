import pandas as pd
from binance.client import Client
from apscheduler.schedulers.background import BackgroundScheduler
import os
from datetime import datetime

client = Client()
csv_path = 'data/raw/btc_1h_lib.csv'

def update_data():
    if not os.path.exists(csv_path):
        print("No existing CSV; run initial fetch first.")
        return
    
    df = pd.read_csv(csv_path)
    last_ts = pd.to_datetime(df['open_time'].max())
    since_str = last_ts.strftime("%d %b, %Y")

    new_klines = client.get_historical_klines(
        'BTCUSDT', Client.KLINE_INTERVAL_1HOUR, 
        f"{pd.to_datetime(last_open, unit='ms').strftime('%d %b, %Y')}"
        )
    
    if new_klines:
        # Append new rows
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_klines[1:])  # Skip header if exists
        print(f"Appended {len(new_klines)} new rows on {datetime.now()}")

# Run in background (non-blocking for app)
scheduler = BackgroundScheduler()
scheduler.add_job(update_data, 'cron', hour=3)  # Daily at 3 AM
scheduler.start()

print("Scheduler started. Press Ctrl+C to exit.")