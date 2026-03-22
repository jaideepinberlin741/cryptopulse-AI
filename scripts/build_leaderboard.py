import json
import glob
import os
from pathlib import Path

import pandas as pd

METRICS_DIR = "models/metrics"

def load_metrics():
    rows = []
    for path in glob.glob(os.path.join(METRICS_DIR, "*.json")):
        with open(path, "r") as f:
            m = json.load(f)

        # Try to infer timeframe/horizon from filename: e.g. btc_15m_1h_xgb_metrics.json
        name = Path(path).name
        parts = name.split("_")
        timeframe, horizon, model_name = None, None, None

        # Very simple pattern: <timeframe>_<horizon>_<model>.json
        if len(parts) >= 3:
            timeframe = parts[0]
            horizon = parts[1]
            model_name = "_".join(parts[2:]).replace(".json", "")

        row = {
            "timeframe": m.get("timeframe", timeframe),
            "horizon": m.get("horizon", horizon),
            "model": m.get("model_type", model_name),
            "test_macro_f1": m.get("test_macro_f1"),
            "test_accuracy": m.get("test_accuracy"),
            "test_rmse": m.get("test_rmse"),
            "test_mae": m.get("test_mae"),
        }
        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_metrics()
    pd.set_option("display.max_columns", None)
    print("\nLEADERBOARD (test metrics):")
    print(
        df.sort_values(
            ["timeframe", "horizon", "model"],
            na_position="last"
        ).reset_index(drop=True)
    )