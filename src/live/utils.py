import pandas as pd
from pathlib import Path
from typing import List, Dict

LIVE_FEATURE_CSV_BY_TF = {
    '15m': 'data/processed/btc_15m_live_features.csv',
    '1h': 'data/processed/btc_1h_live_features.csv',
    '4h': 'data/processed/btc_4h_live_features.csv',
}

def get_last_n_candles(timeframe: str, n: int = 48) -> List[Dict]:
    """Load last N raw OHLC candles for TF from features CSV."""
    csv_path = Path(LIVE_FEATURE_CSV_BY_TF[timeframe])
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path).tail(n)
    return [{'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']}
            for _, row in df.iterrows()]