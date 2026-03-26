"""
xgboost_regression.py
XGBoost regression model for CryptoPulse AI.
Predicts next-horizon return using the same 48x21 features and splits as the classifier.

Target examples:
- For 15m/15m: next 15m close vs current close
- For 15m/1h : next 1h close vs current close
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor
from pathlib import Path

from src.models.train_utils import (
    TrainingConfig,
    load_dataset,
    make_timeseries_splits,
    save_model,
    save_metrics,
    get_model_paths,
)

# Reuse same key configs as classifier
KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h", "1h"),
    ("1h", "4h"),
]

def flatten_features(X: np.ndarray) -> np.ndarray:
    """Flatten sliding windows: (batch, window=48, features=21) → (batch, 1008)."""
    return X.reshape(X.shape[0], -1)

def get_key_configs():
    return [TrainingConfig(timeframe=tf, horizon=hz) for tf, hz in KEY_CONFIGS]

def load_return_target(config: TrainingConfig) -> np.ndarray:
    base_path = Path(config.data_dir)
    prefix = f"btc_{config.timeframe}_featured_"

    y_path = base_path / f"{prefix}labels_y.npy"
    label_cols_path = base_path / f"{prefix}label_cols.npy"

    y = np.load(y_path)
    label_cols_raw = np.load(label_cols_path)
    label_cols = label_cols_raw.astype(str).tolist()

    # pick the right future_return column matching your horizon
    ret_col_name = f"future_return_{config.horizon}"
    try:
        ret_idx = np.where(np.array(label_cols) == ret_col_name)[0][0]
    except (IndexError, ValueError):
        raise ValueError(
            f"Return column '{ret_col_name}' not found. "
            f"Available: {label_cols}"
        )

    return y[:, ret_idx].astype(np.float32)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        f"{split_name}_n_samples": int(len(y_true)),
        f"{split_name}_mse": float(mse),
        f"{split_name}_rmse": rmse,
        f"{split_name}_mae": float(mae),
        f"{split_name}_r2": float(r2),
    }

def save_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    config: TrainingConfig,
    model_name: str,
) -> str:
    """Save regression predictions to CSV for later analysis."""
    df = pd.DataFrame(
        {
            "target_return": y_true,
            "predicted_return": y_pred,
            "error": y_pred - y_true,
        }
    )
    out_path = (
        f"models/predictions/"
        f"{config.timeframe}_{config.horizon}_{model_name}_{split_name}_reg_predictions.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f" ✓ {split_name} regression predictions saved: {out_path}")
    return out_path

def train_xgboost_regressor(
    config: TrainingConfig,
    max_tune_samples: int = 20_000,
) -> tuple:
    """
    XGBoost regression on horizon returns.

    Steps:
    - Load X, y, t from existing dataset.
    - Build a continuous return target from labels (placeholder mapping for now).
    - Tune a small XGBRegressor grid on a capped subset.
    - Fit final model on full train set and evaluate on train/val/test.
    """
    print(f"\n{'=' * 60}")
    print(f" XGBoost REGRESSION: {config.timeframe}/{config.horizon}")
    print(f"{'=' * 60}")

    # 1. Load & build target
    X, y, t, label_cols = load_dataset(config)
    X_flat = flatten_features(X)

    # Use true future_return_<horizon> from labels_y.npy
    y_target = load_return_target(config)
    print(f" Target stats: mean={y_target.mean():.6f}, std={y_target.std():.6f}")

    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y_target[train_idx]
    X_val, y_val = X_flat[val_idx], y_target[val_idx]
    X_test, y_test = X_flat[test_idx], y_target[test_idx]

    print(
        f" Full data: X_train({len(X_train):,}) | "
        f"features={X_flat.shape[1]}"
    )

    # 2. Tuning sample
    if len(X_train) > max_tune_samples:
        n_sample = max_tune_samples
        sample_idx = np.random.choice(len(X_train), n_sample, replace=False)
        X_train_tune = X_train[sample_idx]
        y_train_tune = y_train[sample_idx]
        print(f" Tuning on {n_sample:,} samples (cap)")
    else:
        X_train_tune, y_train_tune = X_train, y_train
        print(" Tuning on FULL training set (below cap)")

    # 3. Small param grid
    param_grid = {
        "n_estimators": [200],
        "max_depth": [4, 6],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0, 3.0],
    }

    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }

    best_score = float("inf")
    best_params = None

    print(" Hyperparameter tuning (regression)...")
    for param_dict in ParameterGrid(param_grid):
        xgb_params = {**base_params, **param_dict}
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train_tune,
            y_train_tune,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        print(
            f" est={param_dict['n_estimators']:<3} "
            f"depth={param_dict['max_depth']:<2} "
            f"lr={param_dict['learning_rate']:<4} "
            f"sub={param_dict['subsample']:<3} "
            f"col={param_dict['colsample_bytree']:<3} "
            f"lam={param_dict['reg_lambda']:<3} "
            f"RMSE={val_rmse:.6f}"
        )

        if val_rmse < best_score:
            best_score = val_rmse
            best_params = xgb_params.copy()

    if best_params is None:
        raise RuntimeError(
            "No best_params found during XGBoost regression tuning; check data and grid."
        )

    print(f"\n Best regression params: {best_params}")
    print(f" Training final regressor on FULL {len(X_train):,} samples...")

    # 4. Final model
    final_model = XGBRegressor(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 5. Evaluate
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    test_pred = final_model.predict(X_test)

    train_metrics = evaluate_regression(y_train, train_pred, "train")
    val_metrics = evaluate_regression(y_val, val_pred, "val")
    test_metrics = evaluate_regression(y_test, test_pred, "test")

    # Save prediction CSVs
    model_name = "xgb_reg"
    train_csv = save_regression_predictions(
        y_train, train_pred, "train", config, model_name
    )
    val_csv = save_regression_predictions(
        y_val, val_pred, "val", config, model_name
    )
    test_csv = save_regression_predictions(
        y_test, test_pred, "test", config, model_name
    )

    all_metrics = {
        "model_type": "XGBoostRegressor",
        "target_description": "rough mapped return from [-2..2] labels",
        "full_train_size": len(X_train),
        "max_tune_samples": max_tune_samples,
        "best_params": best_params,
        "best_val_rmse": float(best_score),
        "train_pred_csv": train_csv,
        "val_pred_csv": val_csv,
        "test_pred_csv": test_csv,
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_model, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test RMSE: {test_metrics['test_rmse']:.6f}")
    print(f" Test predictions: {test_csv}")
    return final_model, all_metrics


def batch_train_key_configs_reg():
    """Train regression model on key timeframe/horizon combinations."""
    configs = get_key_configs()
    results_summary = []

    print(f"\n{'=' * 80}")
    print(" BATCH TRAINING XGBOOST REGRESSION - KEY CONFIGS")
    print(f"{'=' * 80}")

    for i, config in enumerate(configs, 1):
        try:
            print(f"\n[{i:2d}/4] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_xgboost_regressor(config)

            summary = {
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "n_samples": (
                    metrics["test_n_samples"]
                    + metrics["val_n_samples"]
                    + metrics["train_n_samples"]
                ),
                "test_rmse": metrics["test_rmse"],
                "test_mae": metrics["test_mae"],
                "val_rmse": metrics["val_rmse"],
                "best_params": metrics["best_params"],
            }
            results_summary.append(summary)

        except Exception as e:
            print(f" FAILED: {e}")
            results_summary.append(
                {
                    "timeframe": config.timeframe,
                    "horizon": config.horizon,
                    "error": str(e)[:200],
                    "test_rmse": np.nan,
                }
            )

    summary_df = pd.DataFrame(results_summary)
    summary_csv = "models/metrics/xgb_reg_key_configs_summary.csv"
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print("\n REGRESSION SUMMARY TABLE:")
    print(
        summary_df.round(6)
        .sort_values("test_rmse", ascending=True, na_position="last")
    )
    print(f"\n Master regression summary: {summary_csv}")

    return results_summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or "--batch" in sys.argv:
        batch_train_key_configs_reg()
    elif "--help" in sys.argv:
        print(
            """
Usage:
  python -m src.models.xgboost_regression          # Batch key configs (regression)
  python -m src.models.xgboost_regression 15m 15m  # Single TF/horizon
  python -m src.models.xgboost_regression --help   # This help
"""
        )
    else:
        timeframe = sys.argv[1]
        horizon = sys.argv[2] if len(sys.argv) > 2 else "15m"
        cfg = TrainingConfig(timeframe=timeframe, horizon=horizon)
        model, metrics = train_xgboost_regressor(cfg)
        print("\n XGBoost REGRESSION COMPLETE!")