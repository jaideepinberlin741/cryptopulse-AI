"""
Random Forest baseline model for CryptoPulse AI.
Multi-class classification on flattened sliding windows (48x21 → 1008 features).
Light hyperparameter tuning on a capped subsample + full-data final fit.
Restricted to key configs: 15m/15m, 15m/1h, 1h/1h, 1h/4h.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

from train_utils import (
    TrainingConfig, load_dataset, make_timeseries_splits,
    evaluate_classification, save_model, save_metrics,
    get_model_paths,
)


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
    """Return TrainingConfig objects for the selected timeframe/horizon pairs."""
    return [TrainingConfig(timeframe=tf, horizon=hz) for tf, hz in KEY_CONFIGS]


def train_random_forest(
    config: TrainingConfig,
    max_tune_samples: int = 20_000,
) -> tuple:
    """
    Train Random Forest with:
    - Hyperparameter tuning on a capped subset of the training data
    - Final model refit on FULL training data with best hyperparameters
    """
    print(f"\n{'='*60}")
    print(f" Random Forest: {config.timeframe}/{config.horizon}")
    print(f"{'='*60}")

    # 1. Load & prepare
    X, y, t, label_cols = load_dataset(config)
    X_flat = flatten_features(X)
    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y[train_idx]
    X_val, y_val = X_flat[val_idx], y[val_idx]
    X_test, y_test = X_flat[test_idx], y[test_idx]

    print(f" Full data: X_train({len(X_train):,}) | X_flat({X_flat.shape[1]})")

    # 2. TUNING SAMPLE: cap tuning sample size
    if len(X_train) > max_tune_samples:
        n_sample = max_tune_samples
        sample_idx = np.random.choice(len(X_train), n_sample, replace=False)
        X_train_tune = X_train[sample_idx]
        y_train_tune = y_train[sample_idx]
        print(f" Tuning on {n_sample:,} samples (cap)")
    else:
        X_train_tune, y_train_tune = X_train, y_train
        print(" Tuning on FULL training set (below cap)")

    # 3. Small param grid for RF
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [12, 20],
        "max_features": ["sqrt", 0.3],
        "min_samples_leaf": [1, 5],
        "class_weight": ["balanced", None],
    }

    best_score = -1.0
    best_params = None

    print(" Hyperparameter tuning...")
    for param_dict in ParameterGrid(param_grid):
        rf = RandomForestClassifier(
            random_state=config.random_state,
            n_estimators=param_dict["n_estimators"],
            max_depth=param_dict["max_depth"],
            max_features=param_dict["max_features"],
            min_samples_leaf=param_dict["min_samples_leaf"],
            class_weight=param_dict["class_weight"],
            n_jobs=-1,
        )

        rf.fit(X_train_tune, y_train_tune)
        val_pred = rf.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")

        cw_str = str(param_dict["class_weight"])[:8]
        print(
            f"  trees={param_dict['n_estimators']:<3} "
            f"depth={param_dict['max_depth']:<3} "
            f"feat={str(param_dict['max_features']):<4} "
            f"leaf={param_dict['min_samples_leaf']:<2} "
            f"weight={cw_str:<9} F1={val_f1:.4f}"
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_params = param_dict.copy()

    if best_params is None:
        raise RuntimeError("No best_params found during RF tuning; check data and grid.")

    print(f"\n Best params found: {best_params}")
    print(f" Training final RF on FULL {len(X_train):,} samples...")

    final_rf = RandomForestClassifier(
        random_state=config.random_state,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight=best_params["class_weight"],
        n_jobs=-1,
    )
    final_rf.fit(X_train, y_train)

    # 4. Evaluate final model
    train_pred = final_rf.predict(X_train)
    val_pred = final_rf.predict(X_val)
    test_pred = final_rf.predict(X_test)

    train_metrics = evaluate_classification(y_train, train_pred, "train")
    val_metrics = evaluate_classification(y_val, val_pred, "val")
    test_metrics = evaluate_classification(y_test, test_pred, "test")

    all_metrics = {
        "model_type": "RandomForest",
        "full_train_size": len(X_train),
        "max_tune_samples": max_tune_samples,
        "best_params": best_params,
        "best_val_f1": float(best_score),
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    # 5. Save
    model_name = "rf"
    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_rf, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test F1: {test_metrics['test_macro_f1']:.4f}")
    return final_rf, all_metrics


def batch_train_key_configs():
    """Train Random Forest on the 4 key timeframe/horizon combinations."""
    configs = get_key_configs()
    results_summary = []

    print(f"\n{'='*80}")
    print(f" BATCH TRAINING RANDOM FOREST - KEY CONFIGS")
    print(f"{'='*80}")

    for i, config in enumerate(configs, 1):
        try:
            print(f"\n[{i:2d}/4] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_random_forest(config)

            summary = {
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "n_samples": (
                    metrics["test_n_samples"]
                    + metrics["val_n_samples"]
                    + metrics["train_n_samples"]
                ),
                "test_f1": metrics["test_macro_f1"],
                "test_acc": metrics["test_accuracy"],
                "val_f1": metrics["val_macro_f1"],
                "best_params": metrics["best_params"],
            }
            results_summary.append(summary)
        except Exception as e:
            print(f" FAILED: {e}")
            results_summary.append({
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "error": str(e)[:100],
            })

    summary_df = pd.DataFrame(results_summary)
    summary_csv = "models/metrics/rf_key_configs_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n SUMMARY TABLE:")
    print(summary_df.round(4).sort_values("test_f1", ascending=False))
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
            python -m src.models.random_forest              # Batch key configs
            python -m src.models.random_forest 15m 15m      # Single TF/horizon
            python -m src.models.random_forest --help       # This help
        """)
    else:
        timeframe = sys.argv[1]
        horizon = sys.argv[2] if len(sys.argv) > 2 else None
        config = TrainingConfig(timeframe=timeframe, horizon=horizon or "15m")
        model, metrics = train_random_forest(config)
        print("\n Random Forest COMPLETE!")
