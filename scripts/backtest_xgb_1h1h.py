import pandas as pd
import numpy as np

def main():
    clf_path = "models/predictions/1h_1h_xgb_test_predictions.csv"
    reg_path = "models/predictions/1h_1h_xgb_reg_test_reg_predictions.csv"

    clf = pd.read_csv(clf_path)
    reg = pd.read_csv(reg_path)

    df = clf.join(reg, lsuffix="_clf", rsuffix="_reg")

    IDX_TO_DIR = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    df["dir_pred"] = df["y_pred"].map(IDX_TO_DIR)

    df["ret_true"] = df["target_return"]
    df["ret_pred"] = df["predicted_return"]

    H_1H = 0.0058

    def backtest_for_k(k: float):
        THRESH = k * H_1H

        def signal_row(r):
            if r["ret_pred"] > THRESH and r["dir_pred"] > 0:
                return 1
            if r["ret_pred"] < -THRESH and r["dir_pred"] < 0:
                return -1
            return 0

        df["signal"] = df.apply(signal_row, axis=1)
        df["pnl"] = df["signal"] * df["ret_true"]

        trades = df["signal"].ne(0).sum()
        if trades == 0:
            return {"k": k, "THRESH": THRESH, "trades": 0,
                    "mean_pnl": np.nan, "hit_rate": np.nan}

        trade_pnls = df.loc[df["signal"] != 0, "pnl"]
        mean_pnl = trade_pnls.mean()
        hit_rate = (trade_pnls > 0).mean()

        return {"k": k, "THRESH": THRESH, "trades": trades,
                "mean_pnl": mean_pnl, "hit_rate": hit_rate}

    results = [backtest_for_k(k) for k in [0.25, 0.5, 0.75, 1.0]]
    res_df = pd.DataFrame(results)
    print(res_df)

if __name__ == "__main__":
    main()