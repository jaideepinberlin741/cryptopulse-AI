"""
User Story 2.2: Multi-TF Labels + Sliding Windows
Creates final ML-ready datasets for all BTC timeframes.

- Input per timeframe:
    data/processed/btc_<tf>_features.csv

- Output per timeframe:
    data/processed/btc_<tf>_featured_labels_X.npy
    data/processed/btc_<tf>_featured_labels_y.npy
    data/processed/btc_<tf>_featured_labels_t.npy
    data/processed/btc_<tf>_featured_label_cols.npy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Timeframes, horizons, thresholds
# ---------------------------------------------------------------------

TIMEFRAMES: List[str] = ["15m", "1h", "4h"]

# Two prediction horizons per TF (except 1w)
HORIZONS_BY_TF: Dict[str, List[str]] = {
    "15m": ["15m", "1h"],
    "1h": ["1h", "4h"],
    "4h": ["4h", "1d"]
}

# Horizon-specific thresholds on pct return (future_price / price - 1)
HORIZON_THRESHOLDS: Dict[str, float] = {

    "15m": 0.0029,  # 0.29 %
    "1h": 0.0058,   # 0.58 %
    "4h": 0.012,     # 1.2   %
    "1d": 0.036,     # 3.6   %
}


@dataclass
class WindowConfig:
    window_size: int
    step_size: int = 1


@dataclass
class PipelineConfig:
    horizons: List[str]
    window: WindowConfig
    timestamp_col: str = "open_time"
    price_col: str = "close"
    sort_ascending: bool = True
    drop_incomplete: bool = True  # drop rows with NaN future prices


# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------

_TIMEDELTA_MAP = {
    "m": "min",
    "h": "h",
    "d": "D",  
}


def horizon_to_timedelta(h: str) -> pd.Timedelta:
    """Convert '15m,'1h', '4h', '1d' to pd.Timedelta."""
    unit = h[-1]
    value = int(h[:-1])
    if unit not in _TIMEDELTA_MAP:
        raise ValueError(f"Unsupported horizon: {h}")
    return pd.to_timedelta(value, unit=_TIMEDELTA_MAP[unit])


def ensure_datetime_index(
    df: pd.DataFrame,
    timestamp_col: str = "open_time",
) -> pd.DataFrame:
    """
    Handles:
      - ms since epoch (Binance-style)
      - ISO-like strings
    """
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    return df.set_index(timestamp_col).sort_index()


# ---------------------------------------------------------------------
# label logic
# ---------------------------------------------------------------------

# Per-timeframe strong-move thresholds (90th percentile-based)
HORIZON_THRESHOLDS = {
    "15m": 0.0029,   # 0.29 %
    "1h":  0.0058,   # 0.58 %
    "4h":  0.0120,   # 1.20 %
    "1d":  0.0360,   # 3.60 %
}

K_NEUTRAL = 0.1  # eps = 10% of strong threshold for that TF


def classify_direction(
    pct_ret: float,
    prev_dir: int,
    thr: float,
    k_neutral: float = 0.1,
) -> int:
    """
        -2 = Bearish (continuation)
        -1 = Sideways -> Bearish or mild bearish
         0 = Sideways (tiny move)
         1 = Sideways -> Bullish or mild bullish
         2 = Bullish (continuation)
    """
    eps = k_neutral * thr

    if pct_ret > thr:
        base = 1
    elif pct_ret < -thr:
        base = -1
    elif pct_ret > eps:
        base = 0.5
    elif pct_ret < -eps:
        base = -0.5
    else:
        base = 0

    if base == 1:
        return 1 if prev_dir in (0, -1) else 2
    if base == -1:
        return -1 if prev_dir in (0, 1) else -2
    if base == 0.5:
        return 1
    if base == -0.5:
        return -1
    return 0

def assign_trend_labels_from_structure(
    df: pd.DataFrame,
    hh_col: str = "is_higher_high",
    hl_col: str = "is_higher_low",
    lh_col: str = "is_lower_high",
    ll_col: str = "is_lower_low",
    trend_state_col: str = "trend_state",
    swing_window: int = 5,
) -> pd.Series:
    """
    Build 5-class trend labels using swing structure features:

    -2 = Bearish (persistent LH+LL, trend_state < 0)
    -1 = SideBear (weak/early downtrend)
     0 = Neutral (mixed / noisy structure)
     1 = SideBull (weak/early uptrend)
     2 = Bullish (persistent HH+HL, trend_state > 0)
    """
    hh = df[hh_col].astype(int)
    hl = df[hl_col].astype(int)
    lh = df[lh_col].astype(int)
    ll = df[ll_col].astype(int)
    ts = df[trend_state_col].astype(int)

    # rolling counts of structural confirmations
    up_score = (hh + hl).rolling(swing_window, min_periods=1).sum()
    down_score = (lh + ll).rolling(swing_window, min_periods=1).sum()

    up_ratio = up_score / swing_window
    down_ratio = down_score / swing_window

    labels = np.zeros(len(df), dtype=np.int8)

    # thresholds – tune later if needed
    strong_thresh = 0.6   # ≥60% of last N swings aligned
    weak_thresh   = 0.4   # 30–60% = SideBull/SideBear

    # Strong Bullish
    bull_strong = (up_ratio >= strong_thresh) & (up_ratio > down_ratio) & (ts > 0)
    labels[bull_strong.values] = 2

    # Strong Bearish
    bear_strong = (down_ratio >= strong_thresh) & (down_ratio > up_ratio) & (ts < 0)
    labels[bear_strong.values] = -2

    # Weak / grinding up
    bull_weak = (
        (labels == 0)
        & (up_ratio >= weak_thresh)
        & (up_ratio > down_ratio)
    )
    labels[bull_weak] = 1

    # Weak / grinding down
    bear_weak = (
        (labels == 0)
        & (down_ratio >= weak_thresh)
        & (down_ratio > up_ratio)
    )
    labels[bear_weak] = -1

    # Remaining stay 0 (Neutral)
    return pd.Series(labels, index=df.index)

# ---------------------------------------------------------------------
# Label generation for one DataFrame
# ---------------------------------------------------------------------

def add_multi_horizon_labels(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    For each horizon h in cfg.horizons:
      - compute future_price_h, future_return_h, large_move_h
    Then:
      - compute a single trend-based label from structure features
        and reuse it as label_h for all horizons of this TF.
    """
    df = df.copy()

    # 1) Horizon-specific future prices / returns / large-move flags
    for h in cfg.horizons:
        td = horizon_to_timedelta(h)
        thr = HORIZON_THRESHOLDS[h]

        future_price_col = f"future_price_{h}"
        future_return_col = f"future_return_{h}"
        label_col = f"label_{h}"
        large_move_col = f"large_move_{h}"

        future_index = df.index + td
        df[future_price_col] = df[cfg.price_col].reindex(future_index).values
        df[future_return_col] = df[future_price_col] / df[cfg.price_col] - 1.0

        # large-move indicator
        df[large_move_col] = (df[future_return_col].abs() >= thr).astype(np.int8)

    # 2) Trend-based direction label from structure (shared across horizons)
    trend_labels = assign_trend_labels_from_structure(df)

    for h in cfg.horizons:
        df[f"label_{h}"] = trend_labels.astype(np.int8)

    # 3) Drop rows where any future price is NaN (incomplete horizons)
    if cfg.drop_incomplete:
        future_cols = [f"future_price_{h}" for h in cfg.horizons]
        df = df.dropna(subset=future_cols)

    return df

# ---------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------

def make_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_cols: List[str],
    window_cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows for one timeframe.

    Returns:
      X: (num_windows, window_size, num_features)
      y: (num_windows, num_labels)
      t: (num_windows) timestamps at window end
    """
    window = window_cfg.window_size
    step = window_cfg.step_size

    feat = df[feature_cols].values
    lab = df[label_cols].values
    idx = df.index.to_numpy()

    if len(df) < window:
        raise ValueError("Not enough rows to form a single window.")

    num_windows = (len(df) - window) // step + 1
    X = np.empty((num_windows, window, len(feature_cols)), dtype=np.float32)
    y = np.empty((num_windows, lab.shape[1]), dtype=np.int8)
    t = np.empty(num_windows, dtype="datetime64[ns]")

    i = 0
    start = 0
    while start + window <= len(df):
        end = start + window
        X[i] = feat[start:end]
        y[i] = lab[end - 1]
        t[i] = idx[end - 1]
        i += 1
        start += step

    return X, y, t


# ---------------------------------------------------------------------
# End-to-end builder for one CSV
# ---------------------------------------------------------------------

def build_dataset_from_csv(
    csv_path: str,
    cfg: PipelineConfig,
    feature_cols: List[str],
    output_dir: Optional[str] = None,
    prefix: str = "dataset",
) -> Dict[str, np.ndarray]:
    """
    Load CSV (processed), add labels, build sliding-window dataset.

    Args:
      csv_path: e.g. 'data/processed/btc_15m_features.csv'
      cfg:      PipelineConfig
      feature_cols: explicit list of feature columns to use as inputs.
      output_dir: if set, saves X, y, t, label_cols as .npy.
      prefix:  filename prefix for saved arrays.

    Returns:
      dict with {'X', 'y', 't', 'label_cols'}
    """
    df = pd.read_csv(csv_path)
    df = ensure_datetime_index(df, timestamp_col=cfg.timestamp_col)
    if not cfg.sort_ascending:
        df = df.sort_index(ascending=True)

    df = add_multi_horizon_labels(df, cfg)

    label_cols_cls = [f"label_{h}" for h in cfg.horizons]
    label_cols_ret = [f"future_return_{h}" for h in cfg.horizons]
    label_cols_large = [f"large_move_{h}" for h in cfg.horizons]

    label_cols = label_cols_cls + label_cols_ret + label_cols_large

    X, y, t = make_sliding_windows(
        df=df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        window_cfg=cfg.window,
    )

    result = {
        "X": X,
        "y": y,
        "t": t,
        "label_cols": np.array(label_cols),
    }
    # saving code stays the same

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{prefix}labels_X.npy"), X)
        np.save(os.path.join(output_dir, f"{prefix}labels_y.npy"), y)
        np.save(os.path.join(output_dir, f"{prefix}labels_t.npy"), t.astype("datetime64[ns]"))
        np.save(
            os.path.join(output_dir, f"{prefix}label_cols.npy"),
            result["label_cols"],
        )

    return result


# ---------------------------------------------------------------------
# Main: loop all TFs, use HORIZONS_BY_TF
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # All engineered feature columns
    feature_cols = [
       "open", "high", "low", "close", "volume",
        "returns", "hl_range", "hl_pct",
        "volatility", "volatility_short",
        "sma_20", "sma_50", "sma_ratio",
        "rsi", "macd", "macd_signal", "macd_histogram",
        "bb_width", "bb_position",
        "volume_ratio", "price_position",
        # new structural features
        "swing_high", "swing_low",
        "last_swing_high", "last_swing_low",
        "prev_swing_high", "prev_swing_low",
        "is_higher_high", "is_higher_low",
        "is_lower_high", "is_lower_low",
        "trend_state",
    ]

    for tf in TIMEFRAMES:
        csv_path = f"data/processed/btc_{tf}_features.csv"
        if not os.path.exists(csv_path):
            print(f"[WARN] Skipping {tf}: {csv_path} not found")
            continue

        horizons = HORIZONS_BY_TF[tf]

        cfg = PipelineConfig(
            horizons=horizons,
            window=WindowConfig(window_size=48, step_size=1),
            timestamp_col="open_time",
            price_col="close",
            sort_ascending=True,
            drop_incomplete=True,
        )

        prefix = f"btc_{tf}_featured_"

        print(f"[LABEL] {tf}: horizons={horizons}, csv={csv_path}")

        dataset = build_dataset_from_csv(
            csv_path=csv_path,
            cfg=cfg,
            feature_cols=feature_cols,
            output_dir="data/processed",
            prefix=prefix,
        )

        print(
            f"[LABEL] {tf}: X={dataset['X'].shape}, "
            f"y={dataset['y'].shape}, labels={dataset['label_cols']}"
        )
