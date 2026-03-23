"""
Random Forest baseline model for CryptoPulse AI.
Multi-class classification on flattened sliding windows (48x21 → 1008 features).
Light hyperparameter tuning on a capped subsample + full-data final fit.
Optimized for speed: small tuning grid, capped samples, fewer trees during tuning.
Supports KEY_CONFIGS: 15m/15m, 15m/1h, 1h/1h, 1h/4h, 4h/4h, 4h/1d.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.models.train_utils import (
    TrainingConfig,
    load_dataset,
    make_timeseries_splits,
    evaluate_classification,
    save_model,
    save_metrics,
    get_model_paths,
)

KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h",  "1h"),
    ("1h",  "4h"),
    ("4h",  "4h"),
    ("4h",  "1d"),
]

# Max samples used for the final full retrain on large TFs (15m has ~200k rows)
# Set to None to use ALL training data (slower but potentially marginally better)
MAX_FINAL_TRAIN_SAMPLES = 100_000


def flatten_features(X: np.ndarray) -> np.ndarray:
    """Flatten sliding windows: (batch, window=48, features=21) → (batch, 1008)."""
    return X.reshape(X.shape[0], -1)


def get_key_configs():
    """Return TrainingConfig objects for the selected timeframe/horizon pairs."""
    return [TrainingConfig(timeframe=tf, horizon=hz) for tf, hz in KEY_CONFIGS]


def train_random_forest(
    config: TrainingConfig,
    max_tune_samples: int = 10_000,   # lowered from 20k → faster tuning
) -> tuple:
    """
    Train Random Forest with:
    - Fast hyperparameter tuning on a small capped subset (10k rows, 50 trees)
    - Final model refit on FULL (or capped) training data with best params (150 trees)
    - Speed optimizations: small grid, fewer trees for tuning, capped final fit for large TFs
    """
    print(f"\n{'=' * 60}")
    print(f" Random Forest: {config.timeframe}/{config.horizon}")
    print(f"{'=' * 60}")

    # 1. Load & prepare
    X, y, t, label_cols = load_dataset(config)
    X_flat = flatten_features(X)
    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y[train_idx]
    X_val,   y_val   = X_flat[val_idx],   y[val_idx]
    X_test,  y_test  = X_flat[test_idx],  y[test_idx]

    print(f" Full data: X_train({len(X_train):,}) | features={X_flat.shape[1]}")

    # 2. TUNING SAMPLE: cap for speed
    if len(X_train) > max_tune_samples:
        sample_idx = np.random.choice(len(X_train), max_tune_samples, replace=False)
        X_tune = X_train[sample_idx]
        y_tune = y_train[sample_idx]
        print(f" Tuning on {max_tune_samples:,} samples (capped from {len(X_train):,})")
    else:
        X_tune, y_tune = X_train, y_train
        print(f" Tuning on FULL training set ({len(X_train):,} samples, below cap)")

    # 3. SMALL param grid — 6 combos instead of 32
    #    During tuning we use only 50 trees for speed.
    #    Best params are then used for final fit with 150 trees.
    param_grid = {
        "max_depth":        [10, 20],
        "min_samples_leaf": [1, 5],
        "class_weight":     ["balanced", None],
    }

    base_tune_params = {
        "n_estimators": 50,        # fast tuning
        "max_features": "sqrt",    # standard, works well for RF
        "random_state": config.random_state,
        "n_jobs": -1,
    }

    best_score  = -1.0
    best_params = None

    print(f" Hyperparameter tuning ({len(list(ParameterGrid(param_grid)))} combos × 50 trees)...")
    for param_dict in ParameterGrid(param_grid):
        rf = RandomForestClassifier(
            **base_tune_params,
            max_depth=param_dict["max_depth"],
            min_samples_leaf=param_dict["min_samples_leaf"],
            class_weight=param_dict["class_weight"],
        )
        rf.fit(X_tune, y_tune)
        val_pred = rf.predict(X_val)
        val_f1   = f1_score(y_val, val_pred, average="macro")

        cw_str = str(param_dict["class_weight"])[:8]
        print(
            f"  depth={param_dict['max_depth']:<3} "
            f"leaf={param_dict['min_samples_leaf']:<2} "
            f"weight={cw_str:<9} "
            f"F1={val_f1:.4f}"
        )

        if val_f1 > best_score:
            best_score  = val_f1
            best_params = param_dict.copy()

    if best_params is None:
        raise RuntimeError("No best_params found during RF tuning; check data and grid.")

    print(f"\n Best params: {best_params} (val F1={best_score:.4f})")

    # 4. FINAL FIT — 150 trees, best params
    #    Cap final training data for very large TFs (e.g. 15m ~200k rows) to stay fast
    if MAX_FINAL_TRAIN_SAMPLES and len(X_train) > MAX_FINAL_TRAIN_SAMPLES:
        final_idx   = np.random.choice(len(X_train), MAX_FINAL_TRAIN_SAMPLES, replace=False)
        X_train_fin = X_train[final_idx]
        y_train_fin = y_train[final_idx]
        print(
            f" Final fit on {MAX_FINAL_TRAIN_SAMPLES:,} samples "
            f"(capped from {len(X_train):,}) with 150 trees..."
        )
    else:
        X_train_fin, y_train_fin = X_train, y_train
        print(f" Final fit on FULL {len(X_train):,} samples with 150 trees...")

    final_rf = RandomForestClassifier(
        n_estimators=150,            # more trees than tuning, fewer than before
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight=best_params["class_weight"],
        max_features="sqrt",
        random_state=config.random_state,
        n_jobs=-1,
    )
    final_rf.fit(X_train_fin, y_train_fin)

    # 5. Evaluate
    train_pred = final_rf.predict(X_train)
    val_pred   = final_rf.predict(X_val)
    test_pred  = final_rf.predict(X_test)

    train_metrics = evaluate_classification(y_train, train_pred, "train")
    val_metrics   = evaluate_classification(y_val,   val_pred,   "val")
    test_metrics  = evaluate_classification(y_test,  test_pred,  "test")

    all_metrics = {
        "model_type":       "RandomForest",
        "full_train_size":  len(X_train),
        "final_train_size": len(X_train_fin),
        "max_tune_samples": max_tune_samples,
        "tune_n_estimators": 50,
        "final_n_estimators": 150,
        "best_params":      best_params,
        "best_val_f1":      float(best_score),
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    # 6. Save
    model_name  = "rf"
    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_rf, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test F1:  {test_metrics['test_macro_f1']:.4f}")
    print(f" Test Acc: {test_metrics['test_accuracy']:.4f}")
    return final_rf, all_metrics


def batch_train_key_configs():
    """Train Random Forest on all 6 key timeframe/horizon combinations."""
    configs = get_key_configs()
    results_summary = []
    n = len(configs)

    print(f"\n{'=' * 80}")
    print(f" BATCH TRAINING RANDOM FOREST — KEY CONFIGS ({n} pairs)")
    print(f"{'=' * 80}")

    for i, config in enumerate(configs, 1):
        try:
            print(f"\n[{i:2d}/{n}] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_random_forest(config)

            summary = {
                "timeframe":  config.timeframe,
                "horizon":    config.horizon,
                "n_samples":  (
                    metrics["test_n_samples"]
                    + metrics["val_n_samples"]
                    + metrics["train_n_samples"]
                ),
                "test_f1":    metrics["test_macro_f1"],
                "test_acc":   metrics["test_accuracy"],
                "val_f1":     metrics["val_macro_f1"],
                "best_params": metrics["best_params"],
            }
            results_summary.append(summary)

        except Exception as e:
            print(f" FAILED: {e}")
            results_summary.append({
                "timeframe": config.timeframe,
                "horizon":   config.horizon,
                "error":     str(e)[:200],
                "test_f1":   np.nan,
            })

    summary_df  = pd.DataFrame(results_summary)
    summary_csv = "models/metrics/rf_key_configs_summary.csv"
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print("\n SUMMARY TABLE:")
    print(
        summary_df.round(4)
        .sort_values("test_f1", ascending=False, na_position="last")
    )
    print(f"\n Master summary: {summary_csv}")

    return results_summary


# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or "--batch" in sys.argv:
        batch_train_key_configs()
    elif "--help" in sys.argv:
        print("""
Usage:
  python -m src.models.random_forest              # Batch all 6 key configs
  python -m src.models.random_forest 15m 15m      # Single TF/horizon
  python -m src.models.random_forest 1h 4h
  python -m src.models.random_forest --help       # This help
        """)
    else:
        timeframe = sys.argv[1]
        horizon   = sys.argv[2] if len(sys.argv) > 2 else "15m"
        config    = TrainingConfig(timeframe=timeframe, horizon=horizon)
        model, metrics = train_random_forest(config)
        print("\n Random Forest COMPLETE!")