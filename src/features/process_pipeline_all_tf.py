import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os

TIMEFRAMES = ['3m', '5m', '15m', '1h', '4h', '1d', '1w']

def get_window(tf):
    """Adaptive window scaling per timeframe."""
    tf_map = {'3m': 14, '5m': 20, '15m': 24, '1h': 24, '4h': 20, '1d': 14, '1w': 10}
    return tf_map.get(tf, 20)

def engineer_features(df, tf):
    """Complete technical analysis suite."""
    # Basic price/volume features
    df['returns'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
    df['hl_pct'] = (df['high'] - df['low']) / df[['high', 'low']].max(axis=1) * 100
    
    # Adaptive window
    window = get_window(tf)
    
    # Volatility (realized BTC vol index)
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(365 * 24) * 100  # Annualized
    df['volatility_short'] = df['returns'].rolling(window//2).std() * 100
    
    # Trend: SMAs
    df['sma_short'] = SMAIndicator(df['close'], window=window).sma_indicator()
    df['sma_long'] = SMAIndicator(df['close'], window=window*2).sma_indicator()
    df['sma_ratio'] = df['sma_short'] / df['sma_long']
    
    # Momentum: RSI
    df['rsi'] = RSIIndicator(df['close'], window=window).rsi()
    
    # MACD
    macd = MACD(df['close'], window_slow=window*2, window_fast=window)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df['close'], window=window)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price position
    df['price_position'] = (df['close'] - df['sma_long']) / df['sma_long'] * 100
    
    # NaN handling (forward fill → drop)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    return df[['returns', 'hl_range', 'hl_pct', 'volatility', 'volatility_short', 
               'sma_ratio', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
               'bb_width', 'bb_position', 'volume_ratio', 'price_position']]

def process_all_timeframes():
    """User Story 2.1: Feature engineering ALL timeframes."""
    os.makedirs('../../data/processed', exist_ok=True)
    
    for tf in TIMEFRAMES:
        print(f"\n🔄 Processing {tf}...")
        
        # Load raw
        raw_path = f"../../data/raw/btc_{tf}_raw.csv"
        if not os.path.exists(raw_path):
            print(f"❌ Missing {raw_path}")
            continue
            
        df = pd.read_csv(raw_path)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Feature Engineering
        df_features = engineer_features(df, tf)
        
        # Save
        output = f"../../data/processed/btc_{tf}_features.csv"
        df_features.to_csv(output)
        print(f"✅ {len(df_features):,} rows, {len(df_features.columns)} features → {output}")
    
    print("\n🎉 Feature engineering COMPLETE!")

if __name__ == "__main__":
    process_all_timeframes()
