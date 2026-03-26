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
from src.features.process_pipeline_all_tf import (
    engineer_features_all_timeframes,
    engineer_live_features_all,
)

# Logging setup

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
log_dir = os.path.join(BASE_DIR, "logs")
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

# NEW: path setup (must come AFTER logger is defined)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

logger.info(f"BASE_DIR={BASE_DIR}")
logger.info(f"RAW_DIR={RAW_DIR}")
logger.info(f"PROCESSED_DIR={PROCESSED_DIR}")

client = Client()
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

def update_timeframe(tf, max_retries=3):
    csv_path = os.path.join(RAW_DIR, f"btc_{tf}_raw.csv")

    for attempt in range(max_retries):
        try:
            if not os.path.exists(csv_path):
                logger.warning(f"Missing {csv_path}")
                return

            # Load existing raw data
            df = pd.read_csv(csv_path)
            df['open_time'] = pd.to_datetime(df['open_time'])
            last_dt = df['open_time'].max()

            # Fetch new klines (12 fields from Binance)
            klines = client.get_historical_klines('BTCUSDT', tf, limit=10)
            new_df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_vol', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            # Reduce to 6-column raw schema BEFORE any writing
            new_df = new_df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            new_df[['open', 'high', 'low', 'close', 'volume']] = (
                new_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            )
            new_df['open_time'] = pd.to_datetime(new_df['open_time'], unit='ms')

            # Filter only genuinely new bars
            new_rows = new_df[new_df['open_time'] > last_dt]

            if len(new_rows) > 0:
                new_rows.to_csv(csv_path, mode='a', header=False, index=False)
                logger.info(
                    f"✅ {tf}: +{len(new_rows)} bars "
                    f"(latest: {new_rows['open_time'].max()})"
                )
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

    logger.info("📐 Running feature engineering for updated TFs")
    engineer_features_all_timeframes(
        timeframes=TIMEFRAMES,
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )
    logger.info("✅ Cycle + feature update complete")

# Event listener
def job_listener(event):
    job = scheduler.get_job(event.job_id)
    if event.exception:
        logger.error(f"💥 Job '{job.name}' failed: {event.exception}")
    else:
        logger.debug(f"✅ Job '{job.name}' completed")

def update_frequent_tfs():
    frequent_tfs = ["5m", "15m", "1h", "4h"]
    for tf in frequent_tfs:
        update_timeframe(tf)

    engineer_features_all_timeframes(
        timeframes=["15m", "1h", "4h", "1d"],
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )

    # NEW: live 15m/1h/4h features
    engineer_live_features_all(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )

# Multi-frequency scheduler
scheduler = BackgroundScheduler()
scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

scheduler.add_job(update_all, 'cron', hour=3, id='daily_full', name='Daily Full')
scheduler.add_job(
    update_frequent_tfs,
    'interval',
    minutes=3,
    id='frequent_tfs',
    name='Frequent TFs (raw + features)'
)

scheduler.start()
logger.info("""
🚀 Multi-TF Scheduler Active:
├── Daily 3AM: All TFs
└── Every 5min: 5m,15m,1h,4h (live trading)
Logs: ../../logs/scheduler.log
Press Ctrl+C to stop
""")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    scheduler.shutdown()
    logger.info("🛑 Scheduler stopped")