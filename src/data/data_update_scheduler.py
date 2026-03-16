import pandas as pd
import csv
import logging
from logging.handlers import RotatingFileHandler
from binance.client import Client
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
import os
from datetime import datetime
import traceback
import time

# Logging setup
log_dir = '../../logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoScheduler')

client = Client()
TIMEFRAMES = ['3m', '5m', '15m', '1h', '4h', '1d', '1w']

def update_timeframe(tf, max_retries=3):
    csv_path = f'../../data/raw/btc_{tf}_raw.csv'
    
    for attempt in range(max_retries):
        try:
            if not os.path.exists(csv_path):
                logger.warning(f"Missing {csv_path}")
                return
            
            df = pd.read_csv(csv_path)
            df['open_time'] = pd.to_datetime(df['open_time'])  
            last_ts = df['open_time'].max().timestamp() * 1000 
            
            logger.debug(f"{tf}: Last TS = {df['open_time'].max()}")
            
            klines = client.get_historical_klines('BTCUSDT', tf, limit=10)
            new_df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            new_df[['open', 'high', 'low', 'close', 'volume']] = new_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            new_df['open_time'] = pd.to_datetime(new_df['open_time'], unit='ms')
          
            new_rows = new_df[new_df['open_time'].dt.timestamp * 1000 > last_ts]
            
            if len(new_rows) > 0:
                mode = 'a' if os.path.exists(csv_path) else 'w'
                header = not os.path.exists(csv_path)
                new_rows.to_csv(csv_path, mode=mode, header=header, index=False)
                logger.info(f"✅ {tf}: +{len(new_rows)} bars (latest: {new_rows['open_time'].max()})")
                return
            else:
                logger.debug(f"{tf}: No new data")
                return
                
        except Exception as e:
            logger.error(f"❌ {tf} #{attempt+1}: {e}")
            logger.debug(traceback.format_exc())
            if attempt < max_retries - 1:
                time.sleep(30)

def update_all():
    logger.info("🔄 Full update cycle")
    for tf in TIMEFRAMES:
        update_timeframe(tf)
    logger.info("✅ Cycle complete")

# Event listener
def job_listener(event):
    job = scheduler.get_job(event.job_id)
    if event.exception:
        logger.error(f"💥 Job '{job.name}' failed: {event.exception}")
    else:
        logger.debug(f"✅ Job '{job.name}' completed")

# Multi-frequency scheduler
scheduler = BackgroundScheduler()
scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

scheduler.add_job(update_all, 'cron', hour=3, id='daily_full', name='Daily Full')
scheduler.add_job(
    lambda: [update_timeframe(tf) for tf in ['3m', '5m', '15m', '1h', '4h']], 
    'interval', minutes=15, id='frequent_tfs', name='Frequent TFs'
)

scheduler.start()
logger.info("""
🚀 Multi-TF Scheduler Active:
├── Daily 3AM: All 7 TFs
└── Every 15min: 3m,5m,15m,1h,4h (live trading)
Logs: ../../logs/scheduler.log
Press Ctrl+C to stop
""")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    scheduler.shutdown()
    logger.info("🛑 Scheduler stopped")