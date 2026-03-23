"""
Log live next-candle predictions for CryptoPulse AI for all key configs.

For each of:
    (15m,15m), (15m,1h), (1h,1h), (1h,4h)

it:
  - loads latest 48x21 window for the timeframe,
  - runs XGBoost classifier + regressor + large-move,
  - appends one row to models/metrics/live_predictions_<tf>_<hz>.csv.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.models.train_utils import TrainingConfig, get_model_paths
from src.features.label_pipeline import ensure_datetime_index


KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h", "1h"),
    ("1h", "4h"),
    ("4h", "4h"),
    ("4h", "1d"),
]

IDX_TO_CLASS = {
    0: "Bearish",
    1: "SideBear",
    2: "Neutral",
    3: "SideBull",
    4: "Bullish",
}

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "returns", "hl_range", "hl_pct",
    "volatility", "volatility_short",
    "sma_20", "sma_50", "sma_ratio",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "bb_width", "bb_position",
    "volume_ratio", "price_position",
]

FEATURE_CSV_BY_TF: Dict[str, str] = {
    "15m": "data/processed/btc_15m_features.csv",
    "1h": "data/processed/btc_1h_features.csv",
    "4h": "data/processed/btc_4h_features.csv",
}


def load_latest_window(tf: str, window_size: int = 48) -> Tuple[np.ndarray, pd.Timestamp]:
    csv_path = FEATURE_CSV_BY_TF.get(tf)
    if csv_path is None:
        raise ValueError(f"No features CSV configured for timeframe {tf}")

    df = pd.read_csv(csv_path)
    df = ensure_datetime_index(df, timestamp_col="open_time")
    df = df.sort_index(ascending=True)

    if len(df) < window_size:
        raise ValueError(f"Not enough rows ({len(df)}) for one window of size {window_size}.")

    latest = df.iloc[-window_size:]
    X_win = latest[FEATURE_COLS].values.astype(np.float32)
    end_ts = latest.index[-1]
    return X_win.reshape(1, window_size, len(FEATURE_COLS)), end_ts


def flatten_features(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def load_models(config: TrainingConfig):
    clf_model_path, _ = get_model_paths(config, "xgb")
    reg_model_path, _ = get_model_paths(config, "xgb_reg")
    large_model_path, _ = get_model_paths(config, "xgb_large")

    if not Path(clf_model_path).exists():
        raise FileNotFoundError(f"Classifier model not found: {clf_model_path}")

    clf: XGBClassifier = joblib.load(clf_model_path)

    reg = None
    if Path(reg_model_path).exists():
        reg = joblib.load(reg_model_path)

    large = None
    if Path(large_model_path).exists():
        large = joblib.load(large_model_path)

    return clf, reg, large


def get_prediction(tf: str, hz: str) -> Dict:
    cfg = TrainingConfig(timeframe=tf, horizon=hz)
    X_win, end_ts = load_latest_window(tf, window_size=48)
    X_flat = flatten_features(X_win)

    clf, reg, large = load_models(cfg)

    y_proba = clf.predict_proba(X_flat)[0]
    y_pred_idx = int(np.argmax(y_proba))
    class_name = IDX_TO_CLASS.get(y_pred_idx, str(y_pred_idx))
    max_prob = float(np.max(y_proba))

    pred_return = None
    if reg is not None:
        pred_return = float(reg.predict(X_flat)[0])

    prob_large = None
    if large is not None:
        prob_large = float(large.predict_proba(X_flat)[0, 1])

    return {
        "logged_at": datetime.utcnow().isoformat(),
        "window_end": end_ts.isoformat(),
        "timeframe": tf,
        "horizon": hz,
        "class_index": y_pred_idx,
        "class_name": class_name,
        "max_class_prob": max_prob,
        "predicted_return": pred_return,
        "prob_large_move": prob_large,
    }


def append_to_csv(row: dict, path: str):
    path_obj = Path(path)
    os.makedirs(path_obj.parent, exist_ok=True)
    file_exists = path_obj.exists()

    fieldnames = [
        "logged_at",
        "window_end",
        "timeframe",
        "horizon",
        "class_index",
        "class_name",
        "max_class_prob",
        "predicted_return",
        "prob_large_move",
    ]

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    for tf, hz in KEY_CONFIGS:
        try:
            row = get_prediction(tf, hz)
            out_path = f"models/metrics/live_predictions_{tf}_{hz}.csv"
            append_to_csv(row, out_path)
            print(f"[OK] {tf}/{hz} → {out_path}")
        except Exception as e:
            print(f"[FAIL] {tf}/{hz}: {e}")


if __name__ == "__main__":
    main()