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

TIMEFRAMES: List[str] = ["5m", "15m", "1h", "4h", "1d", "1w"]

# Two prediction horizons per TF (except 1w)
HORIZONS_BY_TF: Dict[str, List[str]] = {
    "5m": ["5m", "15m"],
    "15m": ["15m", "1h"],
    "1h": ["1h", "4h"],
    "4h": ["4h", "1d"],
    "1d": ["1d", "1w"],
    "1w": ["1w"],
}

# Horizon-specific thresholds on pct return (future_price / price - 1)
HORIZON_THRESHOLDS: Dict[str, float] = {
    "5m": 0.0017,    # 0.17 %
    "15m": 0.0029,  # 0.29 %
    "1h": 0.0058,   # 0.58 %
    "4h": 0.012,     # 1.2   %
    "1d": 0.036,     # 3.6   %
    "1w": 0.113,     # 11.3   %
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
    "w": "W",
}


def horizon_to_timedelta(h: str) -> pd.Timedelta:
    """Convert '5m','15m,'1h', '4h', '1d', '1w' to pd.Timedelta."""
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
    "5m":  0.0017,   # 0.17 %
    "15m": 0.0029,   # 0.29 %
    "1h":  0.0058,   # 0.58 %
    "4h":  0.0120,   # 1.20 %
    "1d":  0.0360,   # 3.60 %
    "1w":  0.1130,   # 11.30 %
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


# ---------------------------------------------------------------------
# Label generation for one DataFrame
# ---------------------------------------------------------------------

def add_multi_horizon_labels(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    For each horizon H in cfg.horizons, add:

      future_price_H
      future_return_H
      label_H in {-2,-1,0,1,2}
    """
    df = df.copy()

    for h in cfg.horizons:
        td = horizon_to_timedelta(h)
        thr = HORIZON_THRESHOLDS[h]

        future_price_col = f"future_price_{h}"
        future_return_col = f"future_return_{h}"
        label_col = f"label_{h}"

        # Time-based future price: value at t + horizon
        future_index = df.index + td
        df[future_price_col] = df[cfg.price_col].reindex(future_index).values
        df[future_return_col] = df[future_price_col] / df[cfg.price_col] - 1.0

        rets = df[future_return_col].values
        labels = np.zeros_like(rets, dtype=np.int8)

        prev_dir = 0
        for i, r in enumerate(rets):
            if np.isnan(r):
                labels[i] = 0
            else:
                d = classify_direction(r, prev_dir, thr)
                labels[i] = d
                prev_dir = d

        df[label_col] = labels

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
      csv_path: e.g. 'data/processed/btc_5m_features.csv'
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

    label_cols = [f"label_{h}" for h in cfg.horizons]

    X, y, t = make_sliding_windows(
        df=df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        window_cfg=cfg.window,
    )

    result: Dict[str, np.ndarray] = {
        "X": X,
        "y": y,
        "t": t,
        "label_cols": np.array(label_cols),
    }

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
