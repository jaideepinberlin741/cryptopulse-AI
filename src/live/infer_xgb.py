"""
Live next-candle prediction for CryptoPulse AI (XGBoost).

- Supports KEY_CONFIGS:
    (15m,15m), (15m,1h), (1h,1h), (1h,4h)
- For a given timeframe/horizon, loads:
    - XGBoost classifier   (xgb)
    - XGBoost regressor    (xgb_reg)      [optional]
    - XGBoost large-move   (xgb_large)    [optional]
- Uses latest 48x21 window from the corresponding features CSV.
- Prints: direction class, class probabilities, predicted return, and
  probability of a large move.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor

from src.models.train_utils import TrainingConfig, get_model_paths
from src.features.label_pipeline import ensure_datetime_index

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# Must match LABEL_MAP / CLASS_NAMES in xgboost_model.py
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

# Where your features live per timeframe (closed-bar)
FEATURE_CSV_BY_TF = {
    "15m": "data/processed/btc_15m_features.csv",
    "1h":  "data/processed/btc_1h_features.csv",
    "4h":  "data/processed/btc_4h_features.csv",
    "1d":  "data/processed/btc_1d_features.csv",
}

# Live-features overrides
LIVE_FEATURE_CSV_BY_TF = {
    "15m": "data/processed/btc_15m_live_features.csv",
    "1h":  "data/processed/btc_1h_live_features.csv",
    "4h":  "data/processed/btc_4h_live_features.csv",
}


def _get_features_path(tf: str, live: bool = False) -> str:
    """
    Return the path to the features CSV for a timeframe.

    If live=True and a live-features file exists for that TF,
    prefer that; otherwise fall back to the standard features CSV.
    """
    if live:
        live_path = LIVE_FEATURE_CSV_BY_TF.get(tf)
        if live_path is not None and Path(live_path).exists():
            return live_path

    path = FEATURE_CSV_BY_TF.get(tf)
    if path is None:
        raise ValueError(f"No features CSV configured for timeframe {tf}")
    return path

def load_latest_window(tf: str, window_size: int = 48) -> Tuple[np.ndarray, pd.Timestamp]:
    """
    Load the most recent sliding window from the features CSV for a timeframe.

    For 15m, 1h, 4h this will use the live-features file if available so that the
    last row reflects the current, partially formed higher-TF candle.
    """
    use_live = tf in {"15m", "1h", "4h"}
    csv_path = _get_features_path(tf, live=use_live)

    df = pd.read_csv(csv_path)
    df = ensure_datetime_index(df, timestamp_col="open_time")
    df = df.sort_index(ascending=True)

    if len(df) < window_size:
        raise ValueError(
            f"Not enough rows ({len(df)}) for one window of size {window_size} "
            f"from {csv_path}."
        )

    latest = df.iloc[-window_size:]
    X_win = latest[FEATURE_COLS].values.astype(np.float32)
    end_ts = latest.index[-1]

    return X_win.reshape(1, window_size, len(FEATURE_COLS)), end_ts


def flatten_features(X: np.ndarray) -> np.ndarray:
    """(batch, window, features) -> (batch, window*features)."""
    return X.reshape(X.shape[0], -1)


def load_models(config: TrainingConfig) -> Tuple[XGBClassifier, XGBRegressor | None, XGBClassifier | None]:
    """
    Load classifier, regressor, and large-move models using same paths as training.
    Assumes save_model() used joblib/pickle.
    """
    # Classifier
    clf_model_path, _ = get_model_paths(config, "xgb")
    if not Path(clf_model_path).exists():
        raise FileNotFoundError(f"Classifier model not found: {clf_model_path}")
    clf: XGBClassifier = joblib.load(clf_model_path)

    # Regressor (optional)
    reg_model_path, _ = get_model_paths(config, "xgb_reg")
    reg: XGBRegressor | None = None
    if Path(reg_model_path).exists():
        reg = joblib.load(reg_model_path)

    # Large-move classifier (optional)
    large_model_path, _ = get_model_paths(config, "xgb_large")
    large: XGBClassifier | None = None
    if Path(large_model_path).exists():
        large = joblib.load(large_model_path)

    return clf, reg, large

def predict_for_config(timeframe: str, horizon: str) -> dict:
    """
    Core prediction function used by CLI and backend.
    Returns a unified dict with direction, confidence, and optional extras.
    """
    cfg = TrainingConfig(timeframe=timeframe, horizon=horizon)

    X_win, end_ts = load_latest_window(timeframe, window_size=48)
    X_flat = flatten_features(X_win)

    clf, reg, large = load_models(cfg)

    # Direction (5-class)
    y_proba = clf.predict_proba(X_flat)[0]
    y_pred_idx = int(np.argmax(y_proba))
    class_name = IDX_TO_CLASS.get(y_pred_idx, str(y_pred_idx))
    confidence = float(y_proba[y_pred_idx])

    # Optional regressor
    pred_return = None
    if reg is not None:
        pred_return = float(reg.predict(X_flat)[0])

    # Optional large-move classifier
    prob_large = None
    if large is not None:
        prob_large = float(large.predict_proba(X_flat)[0, 1])

    result = {
        "timeframe": timeframe,
        "horizon": horizon,
        "as_of": end_ts.to_pydatetime().isoformat(),
        "direction": class_name,
        "direction_index": y_pred_idx,
        "confidence": confidence,
        "probs": {
            IDX_TO_CLASS[i]: float(p) for i, p in enumerate(y_proba)
        },
        "predicted_return": pred_return,
        "prob_large_move": prob_large,
    }
    return result

def infer_once(timeframe: str, horizon: str) -> None:
    result = predict_for_config(timeframe, horizon)

    print("\n=== CRYPTOPULSE NEXT-CANDLE PREDICTION ===")
    print(f"Timeframe / horizon : {result['timeframe']} / {result['horizon']}")
    print(f"Window end time     : {result['as_of']}")
    print(
        f"Direction class     : {result['direction']} "
        f"(index={result['direction_index']}, conf={result['confidence']:.3f})"
    )
    print("Class probabilities :")
    for cname, p in result["probs"].items():
        print(f"  {cname:8s}: {p:.3f}")

    if result["predicted_return"] is not None:
        r = result["predicted_return"]
        print(f"Predicted return    : {r:.5f}  (~{r*100:.2f}%)")

    if result["prob_large_move"] is not None:
        print(f"P(large move)       : {result['prob_large_move']:.3f}")

    print("==========================================\n")

CHART_TO_CONFIGS = {
    "15m": [("15m", "15m"), ("15m", "1h")],
    "1h":  [("1h", "1h"),  ("1h", "4h")],
    "4h":  [("4h", "4h"),  ("4h", "1d")],
}

def get_predictions_for_chart(chart_tf: str) -> list[dict]:
    """
    Return predictions for the two configs associated with a chart timeframe.
    Example: chart_tf='15m' -> 15m/15m and 15m/1h.
    """
    if chart_tf not in CHART_TO_CONFIGS:
        raise ValueError(f"Unsupported chart_tf={chart_tf}")

    results = []
    for tf, hz in CHART_TO_CONFIGS[chart_tf]:
        results.append(predict_for_config(tf, hz))
    return results

if __name__ == "__main__":
    # CLI usage:
    #   python -m src.live.infer_xgb              -> default 1h/1h
    #   python -m src.live.infer_xgb 15m 1h
    #   python -m src.live.infer_xgb 1h 4h
    if len(sys.argv) == 1:
        tf, hz = "1h", "1h"
    elif len(sys.argv) == 2:
        tf, hz = sys.argv[1], "1h"
    else:
        tf, hz = sys.argv[1], sys.argv[2]

    infer_once(tf, hz)