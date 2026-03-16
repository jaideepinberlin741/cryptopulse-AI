import os
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


TIMEFRAMES = ["3m", "5m", "15m", "1h", "4h", "1d", "1w"]


def engineer_features_for_file(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Create OHLCV + engineered features for a single timeframe CSV.
    """
    df = pd.read_csv(input_csv)

    # Ensure time index
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    df.set_index("open_time", inplace=True)

    # Keep OHLCV as float
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # --- Basic returns and ranges ---
    df["returns"] = df["close"].pct_change()

    df["hl_range"] = df["high"] - df["low"]
    df["hl_pct"] = df["hl_range"] / df["close"] * 100

    df["volatility"] = df["returns"].rolling(24).std() * 100
    df["volatility_short"] = df["returns"].rolling(6).std() * 100

    # --- Trend indicators ---
    sma_short = SMAIndicator(df["close"], window=20)
    sma_long = SMAIndicator(df["close"], window=50)
    df["sma_20"] = sma_short.sma_indicator()
    df["sma_50"] = sma_long.sma_indicator()
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # --- Bollinger Bands derived features ---
    bb = BollingerBands(df["close"], window=20)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()

    df["bb_width"] = (bb_upper - bb_lower) / df["close"] * 100
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    # --- Volume / price positioning ---
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    rolling_low = df["low"].rolling(50).min()
    rolling_high = df["high"].rolling(50).max()
    df["price_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low)

    # Cleanup NaNs / inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    df.reset_index().to_csv(output_csv, index=False)

    print(f"[FEAT] {input_csv} -> {output_csv}, rows={len(df)}")
    print("[FEAT] Columns:", df.columns.tolist())
    return df


def engineer_features_all_timeframes(
    timeframes=None,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> None:
    """
    Run feature engineering for all BTC timeframes.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    for tf in timeframes:
        in_path = os.path.join(raw_dir, f"btc_{tf}_raw.csv")
        out_path = os.path.join(processed_dir, f"btc_{tf}_features.csv")

        if not os.path.exists(in_path):
            print(f"[WARN] Skipping {tf}: raw file not found: {in_path}")
            continue

        engineer_features_for_file(in_path, out_path)


if __name__ == "__main__":
    engineer_features_all_timeframes()
