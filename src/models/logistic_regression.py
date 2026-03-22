"""
Logistic Regression baseline model for CryptoPulse AI.
Multi-class classification on flattened sliding windows (48x21 → 1008 features).
Fast hyperparameter tuning on a capped subsample + (optionally capped) final fit.

Aligned with production KEY_CONFIGS (same as XGBoost / Random Forest):
    (15m, 15m), (15m, 1h), (1h, 1h), (1h, 4h), (4h, 4h), (4h, 1d)

Speed optimizations:
  - max_iter=200 for tuning, 300 for final fit
  - lbfgs solver for 1h/4h, saga for 15m (larger data)
  - param grid: 2 C values × 2 class_weight = 4 combos
  - final fit capped at 100k for 15m TF
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

from src.models.train_utils import (
    TrainingConfig,
    load_dataset,
    make_timeseries_splits,
    evaluate_classification,
    save_model,
    save_metrics,
    get_model_paths,
)

# Aligned with XGBoost / Random Forest
KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h",  "1h"),
    ("1h",  "4h"),
    ("4h",  "4h"),
    ("4h",  "1d"),
]

# Final fit sample cap per base TF
# 15m can have ~200k rows → cap to keep training fast
# 1h/4h are small enough → no cap needed
MAX_FINAL_TRAIN_SAMPLES = {
    "15m": 100_000,
    "1h":  None,
    "4h":  None,
}

# Tune sample cap per base TF
MAX_TUNE_SAMPLES = {
    "15m": 10_000,
    "1h":  10_000,
    "4h":  10_000,
}

# Solver per base TF:
# saga  → scales better for larger datasets (15m)
# lbfgs → faster convergence for smaller datasets (1h, 4h)
SOLVER_BY_TF = {
    "15m": "saga",
    "1h":  "lbfgs",
    "4h":  "lbfgs",
}


def flatten_features(X: np.ndarray) -> np.ndarray:
    """Flatten sliding windows: (batch, window=48, features=21) → (batch, 1008)."""
    return X.reshape(X.shape[0], -1)


def get_key_configs():
    """Return TrainingConfig objects for the 6 production KEY_CONFIGS."""
    return [TrainingConfig(timeframe=tf, horizon=hz) for tf, hz in KEY_CONFIGS]


def train_logistic_regression(config: TrainingConfig) -> tuple:
    """
    Train Logistic Regression with:
    - Fast hyperparameter tuning on a capped subset
    - Reduced max_iter for speed (200 tuning / 300 final)
    - Solver auto-selected per TF size
    - Final fit capped for 15m TF
    """
    print(f"\n{'=' * 60}")
    print(f" Logistic Regression: {config.timeframe}/{config.horizon}")
    print(f"{'=' * 60}")

    tune_cap       = MAX_TUNE_SAMPLES.get(config.timeframe, 10_000)
    final_cap      = MAX_FINAL_TRAIN_SAMPLES.get(config.timeframe)
    solver         = SOLVER_BY_TF.get(config.timeframe, "lbfgs")
    tune_max_iter  = 200
    final_max_iter = 300

    print(f" Solver: {solver} | tune_cap: {tune_cap:,} | final_cap: {final_cap}")

    # 1. Load & prepare
    X, y, t, label_cols = load_dataset(config)
    X_flat = flatten_features(X)
    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y[train_idx]
    X_val,   y_val   = X_flat[val_idx],   y[val_idx]
    X_test,  y_test  = X_flat[test_idx],  y[test_idx]

    print(f" Full data: X_train({len(X_train):,}) | features={X_flat.shape[1]}")

    # 2. TUNING SAMPLE
    if len(X_train) > tune_cap:
        sample_idx = np.random.choice(len(X_train), tune_cap, replace=False)
        X_tune     = X_train[sample_idx]
        y_tune     = y_train[sample_idx]
        print(f" Tuning on {tune_cap:,} samples (capped from {len(X_train):,})")
    else:
        X_tune, y_tune = X_train, y_train
        print(f" Tuning on FULL training set ({len(X_train):,} samples)")

    # 3. PARAM GRID — 4 combos (fast)
    param_grid = {
        "C":            [0.1, 1.0],
        "class_weight": ["balanced", None],
    }

    best_score  = -1.0
    best_params = None

    print(f" Tuning ({len(list(ParameterGrid(param_grid)))} combos, max_iter={tune_max_iter})...")
    for param_dict in ParameterGrid(param_grid):
        pipeline = Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", LogisticRegression(
                C=param_dict["C"],
                class_weight=param_dict["class_weight"],
                solver=solver,
                max_iter=tune_max_iter,
                multi_class="auto",
                n_jobs=-1 if solver == "saga" else None,
                random_state=config.random_state,
            )),
        ])

        pipeline.fit(X_tune, y_tune)
        val_pred = pipeline.predict(X_val)
        val_f1   = f1_score(y_val, val_pred, average="macro")

        cw_str = str(param_dict["class_weight"])[:8]
        print(f"  C={param_dict['C']:<5} weight={cw_str:<9} F1={val_f1:.4f}")

        if val_f1 > best_score:
            best_score  = val_f1
            best_params = param_dict.copy()

    if best_params is None:
        raise RuntimeError(
            "No best_params found during tuning; check data and grid."
        )

    print(f"\n Best params: {best_params} (val F1={best_score:.4f})")

    # 4. FINAL FIT
    if final_cap and len(X_train) > final_cap:
        final_idx   = np.random.choice(len(X_train), final_cap, replace=False)
        X_train_fin = X_train[final_idx]
        y_train_fin = y_train[final_idx]
        print(
            f" Final fit on {final_cap:,} samples "
            f"(capped from {len(X_train):,}), max_iter={final_max_iter}..."
        )
    else:
        X_train_fin, y_train_fin = X_train, y_train
        print(
            f" Final fit on FULL {len(X_train):,} samples, "
            f"max_iter={final_max_iter}..."
        )

    final_pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            C=best_params["C"],
            class_weight=best_params["class_weight"],
            solver=solver,
            max_iter=final_max_iter,
            multi_class="auto",
            n_jobs=-1 if solver == "saga" else None,
            random_state=config.random_state,
        )),
    ])
    final_pipeline.fit(X_train_fin, y_train_fin)

    # 5. Evaluate
    train_pred = final_pipeline.predict(X_train)
    val_pred   = final_pipeline.predict(X_val)
    test_pred  = final_pipeline.predict(X_test)

    train_metrics = evaluate_classification(y_train, train_pred, "train")
    val_metrics   = evaluate_classification(y_val,   val_pred,   "val")
    test_metrics  = evaluate_classification(y_test,  test_pred,  "test")

    all_metrics = {
        "model_type":       "LogisticRegression",
        "solver":           solver,
        "full_train_size":  len(X_train),
        "final_train_size": len(X_train_fin),
        "tune_cap":         tune_cap,
        "final_cap":        final_cap,
        "tune_max_iter":    tune_max_iter,
        "final_max_iter":   final_max_iter,
        "best_params":      best_params,
        "best_val_f1":      float(best_score),
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    # 6. Save
    model_name  = "logreg"
    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_pipeline, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test F1:  {test_metrics['test_macro_f1']:.4f}")
    print(f" Test Acc: {test_metrics['test_accuracy']:.4f}")
    return final_pipeline, all_metrics


def batch_train_key_configs():
    """Train Logistic Regression on all 6 KEY_CONFIGS."""
    configs = get_key_configs()
    results_summary = []
    n = len(configs)

    print(f"\n{'=' * 80}")
    print(f" BATCH TRAINING LOGISTIC REGRESSION — KEY CONFIGS ({n} pairs)")
    print(f"{'=' * 80}")

    for i, config in enumerate(configs, 1):
        try:
            print(f"\n[{i:2d}/{n}] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_logistic_regression(config)

            summary = {
                "timeframe":    config.timeframe,
                "horizon":      config.horizon,
                "n_samples":    (
                    metrics["test_n_samples"]
                    + metrics["val_n_samples"]
                    + metrics["train_n_samples"]
                ),
                "test_f1":      metrics["test_macro_f1"],
                "test_acc":     metrics["test_accuracy"],
                "val_f1":       metrics["val_macro_f1"],
                "best_C":       metrics["best_params"]["C"],
                "class_weight": str(metrics["best_params"]["class_weight"]),
                "solver":       metrics["solver"],
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
    summary_csv = "models/metrics/logreg_key_configs_summary.csv"
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
  python -m src.models.logistic_regression              # Batch all 6 KEY_CONFIGS
  python -m src.models.logistic_regression 15m 15m      # Single TF/horizon
  python -m src.models.logistic_regression 1h 4h
  python -m src.models.logistic_regression 4h 1d
  python -m src.models.logistic_regression --help       # This help
        """)
    else:
        timeframe = sys.argv[1]
        horizon   = sys.argv[2] if len(sys.argv) > 2 else "1h"
        config    = TrainingConfig(timeframe=timeframe, horizon=horizon)
        model, metrics = train_logistic_regression(config)
        print("\n Logistic Regression COMPLETE!")