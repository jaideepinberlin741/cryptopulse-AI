import pandas as pd
import numpy as np

# Same KEY_CONFIGS you use elsewhere
KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h", "1h"),
    ("1h", "4h"),
]

HORIZON_THRESHOLDS = {
    "15m": 0.0029,
    "1h": 0.0058,
    "4h": 0.012,
    "1d": 0.036,
}

IDX_TO_DIR = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}


def backtest_config(tf: str, hz: str, k_values, p_values):
    """
    Combined-signal backtest for one timeframe/horizon:
      - classifier:   {tf}_{hz}_xgb_test_predictions.csv
      - regressor:    {tf}_{hz}_xgb_reg_test_reg_predictions.csv
      - large-move:   {tf}_{hz}_xgb_large_test_predictions.csv
    Returns a DataFrame of results over (k, p0).
    """
    clf_path = f"models/predictions/{tf}_{hz}_xgb_test_predictions.csv"
    reg_path = f"models/predictions/{tf}_{hz}_xgb_reg_test_reg_predictions.csv"
    large_path = f"models/predictions/{tf}_{hz}_xgb_large_test_predictions.csv"

    try:
        clf = pd.read_csv(clf_path)
        reg = pd.read_csv(reg_path)
        large = pd.read_csv(large_path)
    except FileNotFoundError as e:
        print(f"[SKIP] {tf}/{hz}: missing file: {e}")
        return pd.DataFrame()

    # Merge all on index
    df = clf.join(reg, lsuffix="_clf", rsuffix="_reg")
    df = df.join(large[["y_true", "prob_large_move"]], rsuffix="_large")

    # Direction from classifier (remapped indices -> -2..2)
    df["dir_pred"] = df["y_pred"].map(IDX_TO_DIR)

    # True and predicted returns from regressor
    df["ret_true"] = df["target_return"]
    df["ret_pred"] = df["predicted_return"]

    # Large-move probability from large-move model (prob for class 1)
    df["prob_large"] = df["prob_large_move"]

    H = HORIZON_THRESHOLDS[hz]

    results = []
    for k in k_values:
        THRESH = k * H
        for p0 in p_values:
            def signal_row(r):
                # Long condition
                if (
                    r["dir_pred"] > 0
                    and r["ret_pred"] > THRESH
                    and r["prob_large"] > p0
                ):
                    return 1
                # Short condition
                if (
                    r["dir_pred"] < 0
                    and r["ret_pred"] < -THRESH
                    and r["prob_large"] > p0
                ):
                    return -1
                return 0

            df["signal"] = df.apply(signal_row, axis=1)
            df["pnl"] = df["signal"] * df["ret_true"]

            trades = df["signal"].ne(0).sum()
            if trades == 0:
                results.append(
                    {
                        "timeframe": tf,
                        "horizon": hz,
                        "k": k,
                        "THRESH": THRESH,
                        "p0": p0,
                        "trades": 0,
                        "mean_pnl": np.nan,
                        "hit_rate": np.nan,
                    }
                )
                continue

            trade_pnls = df.loc[df["signal"] != 0, "pnl"]
            mean_pnl = trade_pnls.mean()
            hit_rate = (trade_pnls > 0).mean()

            results.append(
                {
                    "timeframe": tf,
                    "horizon": hz,
                    "k": k,
                    "THRESH": THRESH,
                    "p0": p0,
                    "trades": trades,
                    "mean_pnl": mean_pnl,
                    "hit_rate": hit_rate,
                }
            )

    return pd.DataFrame(results)


def main():
    k_values = [0.25, 0.5]     # fraction of horizon threshold
    p_values = [0.6, 0.7, 0.8] # min prob_large_move

    all_results = []
    for tf, hz in KEY_CONFIGS:
        print(f"\n=== Backtest combined signal: {tf}/{hz} ===")
        df_res = backtest_config(tf, hz, k_values, p_values)
        if not df_res.empty:
            print(df_res.round(6))
            all_results.append(df_res)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv("models/metrics/combined_signal_backtest_all.csv", index=False)
        print("\nSaved combined results to models/metrics/combined_signal_backtest_all.csv")
    else:
        print("\nNo results (probably missing prediction files).")


if __name__ == "__main__":
    main()