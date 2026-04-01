import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from binance.client import Client


CONFIG_PAIRS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h", "1h"),
    ("1h", "4h"),
    ("4h", "4h"),
    ("4h", "1d"),
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREDICTIONS_BASE = os.path.join(BASE_DIR, "models", "metrics")

client = Client()


@st.cache_data(ttl=300)
def load_all_predictions():
    """Load predictions for all configured TF/HZ pairs."""
    all_dfs = {}

    for tf, hz in CONFIG_PAIRS:
        path = os.path.join(PREDICTIONS_BASE, f"live_predictions_{tf}_{hz}.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)

            df["timestamp"] = pd.to_datetime(df["window_end"], utc=True)
            df["timeframe"] = df["timeframe"]
            df["horizon"] = df["horizon"]
            df["direction"] = df["class_name"]
            df["confidence"] = pd.to_numeric(df["max_class_prob"], errors="coerce")
            df["predicted_return"] = pd.to_numeric(df["predicted_return"], errors="coerce")
            df["problargemove"] = pd.to_numeric(df["prob_large_move"], errors="coerce")

            df.sort_values("timestamp", inplace=True)
            all_dfs[(tf, hz)] = df
        else:
            st.warning(f"Missing: live_predictions_{tf}_{hz}.csv")

    return all_dfs


@st.cache_data(ttl=300, show_spinner=False)
def fetch_actual_return_cached(timeframe, horizon, pred_timestamp_str):
    """Cached wrapper for actual return lookups."""
    pred_timestamp = pd.Timestamp(pred_timestamp_str).to_pydatetime()
    return fetch_actual_return(client, timeframe, horizon, pred_timestamp)


def fetch_actual_return(client, timeframe, horizon, pred_timestamp):
    """Fetch real close prices from Binance and compute actual return."""
    tf_map = {
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }

    interval = tf_map.get(timeframe)
    if not interval:
        return np.nan

    h_delta = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }.get(horizon, timedelta(hours=1))

    start_ts = int(pred_timestamp.timestamp() * 1000)
    end_ts = int((pred_timestamp + h_delta).timestamp() * 1000)

    try:
        klines = client.get_historical_klines(
            "BTCUSDT",
            interval,
            start_ts,
            end_ts,
            limit=100,
        )

        if len(klines) < 2:
            return np.nan

        close0 = float(klines[0][4])
        close1 = float(klines[-1][4])
        return (close1 / close0) - 1
    except Exception:
        return np.nan


def compute_metrics(df, lookback_months, client):
    """Compute accuracy and cumulative return for a lookback window."""
    if df.empty:
        return {
            "total_trades": 0,
            "correct": 0,
            "accuracy": 0,
            "cumulative_return": 0,
        }

    cutoff = df["timestamp"].max() - timedelta(days=30 * lookback_months)
    recent = df[df["timestamp"] >= cutoff].copy()

    if recent.empty:
        return {
            "total_trades": 0,
            "correct": 0,
            "accuracy": 0,
            "cumulative_return": 0,
        }

    recent["actual_return"] = recent.apply(
        lambda row: fetch_actual_return_cached(
            row["timeframe"],
            row["horizon"],
            row["timestamp"].isoformat(),
        ),
        axis=1,
    )

    recent["correct"] = (
        np.sign(recent["predicted_return"]) == np.sign(recent["actual_return"])
    )

    recent.loc[recent["predicted_return"] == 0, "correct"] = (
        recent.loc[recent["predicted_return"] == 0, "actual_return"].abs() < 0.001
    )

    total = len(recent)
    correct = recent["correct"].sum()
    accuracy = (correct / total * 100) if total > 0 else 0
    cum_pnl = (1 + recent["actual_return"].fillna(0)).prod() - 1

    return {
        "total_trades": total,
        "correct": int(correct),
        "accuracy": round(accuracy, 1),
        "cumulative_return": round(cum_pnl * 100, 2),
    }


def render_history_tab():
    st.header("📊 Historical Performance")

    all_preds = load_all_predictions()
    if not all_preds:
        st.warning("No prediction files found. Run data scheduler + log_predictions.py")
        return

    selected_pairs = st.multiselect(
        "Select TF/Horizon pairs",
        [f"{tf}/{hz}" for tf, hz in CONFIG_PAIRS],
        default=[f"{tf}/{hz}" for tf, hz in CONFIG_PAIRS[:2]],
    )

    if not selected_pairs:
        st.info("Select at least one TF/Horizon pair.")
        return

    lookback = st.selectbox("Lookback", ["1 month", "3 months", "6 months"])
    months = {"1 month": 1, "3 months": 3, "6 months": 6}[lookback]

    metrics_data = []
    for pair_str in selected_pairs:
        tf, hz = pair_str.split("/")
        df = all_preds.get((tf, hz), pd.DataFrame())
        metrics = compute_metrics(df, months, client)

        metrics_data.append(
            {
                "Pair": f"{tf}/{hz}",
                "Trades": metrics["total_trades"],
                "Correct": f"{metrics['correct']}/{metrics['total_trades']}",
                "Accuracy": f"{metrics['accuracy']}%",
                "Return": f"{metrics['cumulative_return']}%",
            }
        )

    st.subheader("Metrics")
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    if st.checkbox("Show recent trades table"):
        parsed_pairs = [tuple(pair.split("/")) for pair in selected_pairs]

        frames = [
            all_preds[(tf, hz)]
            for tf, hz in parsed_pairs
            if (tf, hz) in all_preds and not all_preds[(tf, hz)].empty
        ]

        if not frames:
            st.info("No recent trades available for the selected TF/Horizon pairs.")
            return

        recent_all = pd.concat(frames, ignore_index=True)
        recent_all = recent_all.sort_values("timestamp")

        cutoff = recent_all["timestamp"].max() - timedelta(days=30 * months)
        recent = recent_all[recent_all["timestamp"] >= cutoff].tail(20).copy()

        if recent.empty:
            st.info("No trades found in the selected lookback window.")
            return

        recent["actual_return"] = recent.apply(
            lambda row: fetch_actual_return_cached(
                row["timeframe"],
                row["horizon"],
                row["timestamp"].isoformat(),
            ),
            axis=1,
        )

        recent["pair"] = recent["timeframe"].astype(str) + "/" + recent["horizon"].astype(str)
        recent["direction"] = recent["direction"].astype(str).str[:4]

        recent["correct"] = (
            np.sign(recent["predicted_return"]) == np.sign(recent["actual_return"])
        )

        recent.loc[recent["predicted_return"] == 0, "correct"] = (
            recent.loc[recent["predicted_return"] == 0, "actual_return"].abs() < 0.001
        )

        recent = recent[
            [
                "timestamp",
                "pair",
                "direction",
                "predicted_return",
                "actual_return",
                "correct",
            ]
        ].round(4)

        recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        recent["correct"] = recent["correct"].map({True: "✅", False: "❌"})

        st.dataframe(recent, use_container_width=True)