import os
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def add_structure_features(df: pd.DataFrame, swing_window: int = 5) -> pd.DataFrame:
    """
    Add basic price-structure features:
    - swing_high / swing_low flags
    - higher_high / higher_low / lower_high / lower_low
    - simple trend_state over last N swings
    """
    # Local swings using rolling max/min
    df["swing_high"] = (
        df["high"]
        == df["high"].rolling(swing_window, center=True).max()
    ).astype(int)

    df["swing_low"] = (
        df["low"]
        == df["low"].rolling(swing_window, center=True).min()
    ).astype(int)

    # Carry forward last swing prices
    df["last_swing_high"] = df["high"].where(df["swing_high"] == 1).ffill()
    df["last_swing_low"] = df["low"].where(df["swing_low"] == 1).ffill()

    # Previous swing prices
    df["prev_swing_high"] = df["last_swing_high"].shift(1)
    df["prev_swing_low"] = df["last_swing_low"].shift(1)

    # Structural relationships
    df["is_higher_high"] = (
        (df["last_swing_high"] > df["prev_swing_high"])
    ).astype(int)

    df["is_higher_low"] = (
        (df["last_swing_low"] > df["prev_swing_low"])
    ).astype(int)

    df["is_lower_high"] = (
        (df["last_swing_high"] < df["prev_swing_high"])
    ).astype(int)

    df["is_lower_low"] = (
        (df["last_swing_low"] < df["prev_swing_low"])
    ).astype(int)

    # Simple trend state over last N swings
    swing_trend = (
        df["is_higher_high"] + df["is_higher_low"]
        - df["is_lower_high"] - df["is_lower_low"]
    ).rolling(5).sum()

    df["trend_state"] = 0
    df.loc[swing_trend > 0, "trend_state"] = 1   # swing up
    df.loc[swing_trend < 0, "trend_state"] = -1  # swing down

    return df


def engineer_features_for_file(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Create OHLCV + engineered features for a single timeframe CSV.
    """
    df = pd.read_csv(input_csv)

    # Ensure time index
    df["open_time"] = pd.to_datetime(df["open_time"], format="mixed", utc=True)
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

    # --- Structural features: swings & HH/HL vs LL/LH ---
    df = add_structure_features(df)

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
    Run feature engineering for all BTC timeframes (closed-bar features).
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


# ============= LIVE FEATURES HELPERS =============

def _build_partial_from_lower_tf(
    lower_tf_csv: str,
    resample_rule: str,
    lookback_bars: int = 500,
) -> pd.DataFrame:
    """
    Generic: build higher-TF OHLCV from lower-TF raw data, including partial bar.

    lower_tf_csv   : path to raw lower-TF CSV (e.g., 5m or 15m)
    resample_rule  : pandas offset alias for target TF ("15T", "1H", "4H")
    """
    if not os.path.exists(lower_tf_csv):
        raise FileNotFoundError(lower_tf_csv)

    df = pd.read_csv(lower_tf_csv)
    df["open_time"] = pd.to_datetime(df["open_time"], format="mixed", utc=True)
    df = df.sort_values("open_time")

    if lookback_bars is not None and len(df) > lookback_bars:
        df = df.iloc[-lookback_bars:]

    df.set_index("open_time", inplace=True)
    ohlcv = df[["open", "high", "low", "close", "volume"]].astype(float)

    out = pd.DataFrame()
    out["open"] = ohlcv["open"].resample(resample_rule).first()
    out["high"] = ohlcv["high"].resample(resample_rule).max()
    out["low"] = ohlcv["low"].resample(resample_rule).min()
    out["close"] = ohlcv["close"].resample(resample_rule).last()
    out["volume"] = ohlcv["volume"].resample(resample_rule).sum()

    out.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return out


def _add_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as for training."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()

    df["hl_range"] = df["high"] - df["low"]
    df["hl_pct"] = df["hl_range"] / df["close"] * 100

    df["volatility"] = df["returns"].rolling(24).std() * 100
    df["volatility_short"] = df["returns"].rolling(6).std() * 100

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

    bb = BollingerBands(df["close"], window=20)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()

    df["bb_width"] = (bb_upper - bb_lower) / df["close"] * 100
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    rolling_low = df["low"].rolling(50).min()
    rolling_high = df["high"].rolling(50).max()
    df["price_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low)

    # --- Structural features: swings & HH/HL vs LL/LH ---
    df = add_structure_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df


def engineer_live_features_all(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> None:
    """
    Build live higher-TF features:

    - 15m live from 5m raw
    - 1h  live from 15m raw
    - 4h  live from 1h raw
    """
    os.makedirs(processed_dir, exist_ok=True)

    # 15m live from 5m
    df15 = _build_partial_from_lower_tf(
        lower_tf_csv=os.path.join(raw_dir, "btc_5m_raw.csv"),
        resample_rule="15T",
    )
    df15 = _add_standard_features(df15)
    df15.reset_index().rename(columns={"index": "open_time"}).to_csv(
        os.path.join(processed_dir, "btc_15m_live_features.csv"), index=False
    )

    # 1h live from 15m
    df1h = _build_partial_from_lower_tf(
        lower_tf_csv=os.path.join(raw_dir, "btc_15m_raw.csv"),
        resample_rule="1H",
    )
    df1h = _add_standard_features(df1h)
    df1h.reset_index().rename(columns={"index": "open_time"}).to_csv(
        os.path.join(processed_dir, "btc_1h_live_features.csv"), index=False
    )

    # 4h live from 1h
    df4h = _build_partial_from_lower_tf(
        lower_tf_csv=os.path.join(raw_dir, "btc_1h_raw.csv"),
        resample_rule="4H",
    )
    df4h = _add_standard_features(df4h)
    df4h.reset_index().rename(columns={"index": "open_time"}).to_csv(
        os.path.join(processed_dir, "btc_4h_live_features.csv"), index=False
    )

    print("[LIVE FEAT] Updated live 15m/1h/4h feature files.")

if __name__ == "__main__":
    engineer_features_all_timeframes()