import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os

def engineer_features(input_csv, output_csv):
    """Technical indicators using 'ta' library."""
    # Load raw BTC
    df = pd.read_csv("../../data/raw/btc_1h_lib.csv")
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # Basic features
    df['returns'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
    df['volatility'] = df['returns'].rolling(24).std() * 100
    
    # Indicators
    df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    # NaN handling
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    # Save
    os.makedirs('../../data/processed', exist_ok=True)
    df.to_csv(output_csv)
    print(f"Saved {len(df)} feature rows to {output_csv}")
    print(df.columns.tolist())  # Verify features
    return df

if __name__ == "__main__":
    engineer_features('btc_1h_lib.csv', '../../data/processed/btc_features_1h.csv')
