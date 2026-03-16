import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os

# --- START: Robust Path Generation ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
raw_data_dir = os.path.join(project_root, 'data', 'raw')
processed_data_dir = os.path.join(project_root, 'data', 'processed')
# --- END: Robust Path Generation ---

TIMEFRAMES = ['3m', '5m', '15m', '1h', '4h', '1d', '1w']

def get_window(tf):
    # ... (function remains the same)
    tf_map = {'3m': 14, '5m': 20, '15m': 24, '1h': 24, '4h': 20, '1d': 14, '1w': 10}
    return tf_map.get(tf, 20)

def engineer_features(df, tf):
    # ... (function remains the same)
    df['returns'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
    df['hl_pct'] = (df['high'] - df['low']) / df[['high', 'low']].max(axis=1) * 100
    window = get_window(tf)
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(365 * 24) * 100
    df['volatility_short'] = df['returns'].rolling(window//2).std() * 100
    df['sma_short'] = SMAIndicator(df['close'], window=window).sma_indicator()
    df['sma_long'] = SMAIndicator(df['close'], window=window*2).sma_indicator()
    df['sma_ratio'] = df['sma_short'] / df['sma_long']
    df['rsi'] = RSIIndicator(df['close'], window=window).rsi()
    macd = MACD(df['close'], window_slow=window*2, window_fast=window)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    bb = BollingerBands(df['close'], window=window)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['volume_sma'] = df['volume'].rolling(window).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['price_position'] = (df['close'] - df['sma_long']) / df['sma_long'] * 100
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df[['returns', 'hl_range', 'hl_pct', 'volatility', 'volatility_short', 
               'sma_ratio', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
               'bb_width', 'bb_position', 'volume_ratio', 'price_position']]

def process_all_timeframes():
    os.makedirs(processed_data_dir, exist_ok=True)
    for tf in TIMEFRAMES:
        print(f"\n🔄 Processing {tf}...")
        raw_path = os.path.join(raw_data_dir, f"btc_{tf}_raw.csv")
        if not os.path.exists(raw_path):
            print(f"❌ Missing {raw_path}")
            continue
        df = pd.read_csv(raw_path)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df_features = engineer_features(df, tf)
        output_path = os.path.join(processed_data_dir, f"btc_{tf}_features.csv")
        df_features.to_csv(output_path)
        print(f"✅ {len(df_features):,} rows, {len(df_features.columns)} features → {output_path}")
    print("\n🎉 Feature engineering COMPLETE!")

if __name__ == "__main__":
    process_all_timeframes()