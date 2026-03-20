"""
Logistic Regression baseline model for CryptoPulse AI.
Multi-class classification on flattened sliding windows (48x21 → 1008 features).
Fast hyperparameter tuning on a capped subsample + (optionally capped) final fit.
Batch training for ALL 11 timeframe/horizon combinations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

from train_utils import (
    TrainingConfig, load_dataset, make_timeseries_splits,
    evaluate_classification, save_model, save_metrics,
    get_model_paths, get_all_configs
)


def flatten_features(X: np.ndarray) -> np.ndarray:
    """Flatten sliding windows: (batch, window=48, features=21) → (batch, 1008)."""
    return X.reshape(X.shape[0], -1)


def train_logistic_regression(
    config: TrainingConfig,
    max_tune_samples: int = 10_000,   # cap for tuning speed (default for non-5m)
) -> tuple:
    """
    Train Logistic Regression with:
    - Fast hyperparameter tuning on a capped subset of the training data
    - Final model refit on (optionally capped) training data with best hyperparameters
    """
    print(f"\n{'='*60}")
    print(f" Logistic Regression: {config.timeframe}/{config.horizon}")
    print(f"{'='*60}")

    # 0. Special-case ultra-dense 5m timeframes
    is_5m = config.timeframe == "5m"
    # Stricter caps for 5m; lighter for others
    if is_5m:
        tune_cap = 5_000
        final_cap = 200_000   # cap final fit for 5m
        final_max_iter = 300
    else:
        tune_cap = max_tune_samples
        final_cap = None      # no cap for final fit
        final_max_iter = 500

    # 1. Load & prepare
    X, y, t, label_cols = load_dataset(config)
    X_flat = flatten_features(X)
    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y[train_idx]
    X_val, y_val = X_flat[val_idx], y[val_idx]
    X_test, y_test = X_flat[test_idx], y[test_idx]

    print(f" Full data: X_train({len(X_train):,}) | X_flat({X_flat.shape[1]})")

    # 2. FAST TUNING: cap tuning sample size
    if len(X_train) > tune_cap:
        n_sample = tune_cap
        sample_idx = np.random.choice(len(X_train), n_sample, replace=False)
        X_train_tune = X_train[sample_idx]
        y_train_tune = y_train[sample_idx]
        print(f" Tuning on {n_sample:,} samples (cap, timeframe={config.timeframe})")
    else:
        X_train_tune, y_train_tune = X_train, y_train
        print(" Tuning on FULL training set (below cap)")

    # 3. Smaller/faster grid: only tune C and class_weight
    #    Fix solver, max_iter, and multi_class for speed + stability
    fixed_solver = "saga"        # good for larger, high-dimensional problems [web:31][web:23]
    fixed_max_iter = 500         # tuning max_iter (modest); final_max_iter may differ

    fixed_multi_class = "auto"

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0],
        "classifier__class_weight": ["balanced", None],
    }

    best_score = -1.0
    best_model = None
    best_params = None

    # 4. Grid search (FAST)
    print(" Hyperparameter tuning...")
    for param_dict in ParameterGrid(param_grid):
        # Extract LR params from pipeline-style keys
        grid_params = {k.split("__")[-1]: v for k, v in param_dict.items()}

        # Compose full LR params with fixed choices for tuning
        lr_params = {
            "solver": fixed_solver,
            "max_iter": fixed_max_iter,
            "multi_class": fixed_multi_class,
            **grid_params,
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                random_state=config.random_state,
                **lr_params
            )),
        ])

        pipeline.fit(X_train_tune, y_train_tune)
        val_pred = pipeline.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")

        cw_str = str(lr_params["class_weight"])[:8]
        print(f"  C={lr_params['C']:<5} weight={cw_str:<9} F1={val_f1:.4f}")

        if val_f1 > best_score:
            best_score = val_f1
            best_model = pipeline
            best_params = lr_params.copy()

    # 5. FINAL MODEL: (optionally capped) full dataset + best params
    #    For speed, override max_iter for final fit, and optionally subsample 5m TFs.
    if best_params is None:
        raise RuntimeError("No best_params found during tuning; check data and grid.")

    best_params["max_iter"] = final_max_iter  # enforce cheaper final optimization

    if final_cap is not None and len(X_train) > final_cap:
        final_idx = np.random.choice(len(X_train), final_cap, replace=False)
        X_train_final = X_train[final_idx]
        y_train_final = y_train[final_idx]
        print(
            f"\n Best params found: {best_params}"
            f"\n Final fit on subsample of {final_cap:,} (from {len(X_train):,}) samples..."
        )
    else:
        X_train_final = X_train
        y_train_final = y_train
        print(
            f"\n Best params found: {best_params}"
            f"\n Training final model on FULL {len(X_train):,} samples..."
        )

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=config.random_state,
            **best_params
        )),
    ])
    final_pipeline.fit(X_train_final, y_train_final)

    # 6. Evaluate final model
    train_pred = final_pipeline.predict(X_train)
    val_pred = final_pipeline.predict(X_val)
    test_pred = final_pipeline.predict(X_test)

    train_metrics = evaluate_classification(y_train, train_pred, "train")
    val_metrics = evaluate_classification(y_val, val_pred, "val")
    test_metrics = evaluate_classification(y_test, test_pred, "test")

    all_metrics = {
        "model_type": "LogisticRegression",
        "full_train_size": len(X_train),
        "tune_cap": tune_cap,
        "final_cap": final_cap,
        "best_params": best_params,
        "best_val_f1": float(best_score),
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    # 7. Save
    model_name = "logreg"
    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_pipeline, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test F1: {test_metrics['test_macro_f1']:.4f}")
    return final_pipeline, all_metrics


def batch_train_all_timeframes():
    """Train Logistic Regression on ALL 11 timeframe/horizon combinations."""
    all_configs = get_all_configs()
    results_summary = []

    print(f"\n{'='*80}")
    print(f" BATCH TRAINING LOGISTIC REGRESSION - ALL 11 TIMEFRAMES")
    print(f"{'='*80}")

    for i, config in enumerate(all_configs, 1):
        try:
            print(f"\n[{i:2d}/11] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_logistic_regression(config)

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
                "best_C": metrics["best_params"]["C"],
                "class_weight": str(metrics["best_params"]["class_weight"]),
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
    summary_csv = "models/metrics/logreg_all_timeframes_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n SUMMARY TABLE:")
    print(summary_df.round(4).sort_values("test_f1", ascending=False))
    print(f"\n Master summary: {summary_csv}")

    return results_summary


# COMPLETE CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or "--batch" in sys.argv:
        batch_train_all_timeframes()
    elif "--help" in sys.argv:
        print("""
            Usage:
            python -m src.models.logistic_regression          # Batch ALL 11 TFs
            python -m src.models.logistic_regression 1h 4h    # Single TF/horizon
            python -m src.models.logistic_regression --help   # This help
        """)
    else:
        timeframe = sys.argv[1]
        horizon = sys.argv[2] if len(sys.argv) > 2 else None
        config = TrainingConfig(timeframe=timeframe, horizon=horizon or "4h")
        model, metrics = train_logistic_regression(config)
        print("\n Logistic Regression COMPLETE!")
