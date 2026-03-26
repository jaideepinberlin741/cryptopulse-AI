"""
xgb_large_move.py
XGBoost binary classifier for large-move events in CryptoPulse AI.

For each timeframe/horizon config, it predicts:
    large_move_horizon = 1 if |future_return_horizon| >= HORIZON_THRESHOLDS[horizon] else 0

It reuses:
  - Sliding-window X features (48x21 → 1008).
  - label_cols/y that now include large_move_<h> columns.
  - The same time-series splits and TrainingConfig as other models.

Outputs:
  - models/xgb_large_move/...  (pickled models)
  - models/metrics/...         (JSON metrics)
  - models/predictions/...     (CSV with probs, labels, etc.)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.train_utils import (
    TrainingConfig,
    load_dataset,            # still returns X, y_selected, t, label_cols (we'll load large_move separately)
    make_timeseries_splits,
    save_model,
    save_metrics,
    get_model_paths,
)

# Key configs to train
KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h", "1h"),
    ("1h", "4h"),
]


def flatten_features(X: np.ndarray) -> np.ndarray:
    """(batch, window=48, features=21) → (batch, 1008)."""
    return X.reshape(X.shape[0], -1)


def get_key_configs() -> List[TrainingConfig]:
    return [TrainingConfig(timeframe=tf, horizon=hz) for tf, hz in KEY_CONFIGS]


# ---------- Helper: load large_move label from labels_y ----------

def load_large_move_target(config: TrainingConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Load the binary large_move_<horizon> target directly from labels_y.npy.

    Assumes your label pipeline saved, in order:
      label_<h> for each horizon,
      future_return_<h> for each horizon,
      large_move_<h> for each horizon.

    So label_cols includes e.g.:
      ['label_15m', 'label_1h', 'future_return_15m', 'future_return_1h',
       'large_move_15m', 'large_move_1h']
    """
    base_path = Path(config.data_dir)
    prefix = f"btc_{config.timeframe}_featured_"

    y_path = base_path / f"{prefix}labels_y.npy"
    label_cols_path = base_path / f"{prefix}label_cols.npy"

    y_full = np.load(y_path)
    label_cols_raw = np.load(label_cols_path)

    if label_cols_raw.dtype.kind in ["S", "U"]:
        label_cols = label_cols_raw.astype(str).tolist()
    else:
        label_cols = label_cols_raw.tolist()

    target_name = f"large_move_{config.horizon}"
    try:
        idx = np.where(np.array(label_cols) == target_name)[0][0]
    except (IndexError, ValueError):
        raise ValueError(
            f"Binary target '{target_name}' not found. Available: {label_cols}"
        )

    y_bin = y_full[:, idx].astype(np.int8)
    return y_bin, label_cols


# ---------- Evaluation / artifacts ----------

def plot_confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Large-move Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NoMove", "LargeMove"],
        yticklabels=["NoMove", "LargeMove"],
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ✓ Confusion matrix saved: {save_path}")


def evaluate_binary_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    config: TrainingConfig,
    model_name: str,
) -> Dict:
    metrics = {
        f"{split_name}_n_samples": int(len(y_true)),
        f"{split_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{split_name}_f1": float(f1_score(y_true, y_pred)),
        f"{split_name}_precision": float(precision_score(y_true, y_pred)),
        f"{split_name}_recall": float(recall_score(y_true, y_pred)),
    }

    # AUC if both classes present
    if len(np.unique(y_true)) == 2:
        metrics[f"{split_name}_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics[f"{split_name}_auc"] = np.nan

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_large_move": y_prob,
            "correct": (y_true == y_pred).astype(int),
        }
    )
    pred_csv = (
        f"models/predictions/"
        f"{config.timeframe}_{config.horizon}_{model_name}_{split_name}_predictions.csv"
    )
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    pred_df.to_csv(pred_csv, index=False)
    metrics[f"{split_name}_pred_csv"] = pred_csv
    print(f" ✓ {split_name} predictions saved: {pred_csv} ({len(pred_df)} samples)")

    # Confusion matrix
    cm_path = (
        f"models/metrics/"
        f"{config.timeframe}_{config.horizon}_{model_name}_{split_name}_cm.png"
    )
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plot_confusion_matrix_binary(
        y_true,
        y_pred,
        cm_path,
        title=f"CM LargeMove {config.timeframe}/{config.horizon} {split_name}",
    )
    metrics[f"{split_name}_cm_path"] = cm_path

    # Classification report (for debugging / logs only)
    report = classification_report(
        y_true, y_pred, target_names=["NoMove", "LargeMove"], output_dict=True
    )
    metrics[f"{split_name}_support_0"] = report["NoMove"]["support"]
    metrics[f"{split_name}_support_1"] = report["LargeMove"]["support"]

    return metrics


# ---------- Training ----------

def train_xgb_large_move(
    config: TrainingConfig,
    max_tune_samples: int = 20_000,
) -> Tuple[XGBClassifier, Dict]:
    """
    Binary XGBoost to predict large-move events for a given timeframe/horizon.

    y_binary = 1 if |future_return_h| >= threshold_h else 0.
    """
    print(f"\n{'=' * 60}")
    print(f" XGB LARGE MOVE: {config.timeframe}/{config.horizon}")
    print(f"{'=' * 60}")

    # Load features via existing loader (classification label here is not used)
    X, _, t, _ = load_dataset(config)
    X_flat = flatten_features(X)

    # Load binary large-move label
    y_bin, label_cols = load_large_move_target(config)
    print(f" Available label_cols: {label_cols}")
    print(f" Large-move classes: {np.unique(y_bin)} (0=no, 1=yes)")

    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y_bin[train_idx]
    X_val, y_val = X_flat[val_idx], y_bin[val_idx]
    X_test, y_test = X_flat[test_idx], y_bin[test_idx]

    print(
        f" Full data: X_train({len(X_train):,}) | features={X_flat.shape[1]}"
    )
    print(
        f" Positives in train/val/test: "
        f"{y_train.sum()}/{y_val.sum()}/{y_test.sum()}"
    )

    # Tuning sample cap
    if len(X_train) > max_tune_samples:
        n_sample = max_tune_samples
        sample_idx = np.random.choice(len(X_train), n_sample, replace=False)
        X_train_tune = X_train[sample_idx]
        y_train_tune = y_train[sample_idx]
        print(f" Tuning on {n_sample:,} samples (cap)")
    else:
        X_train_tune, y_train_tune = X_train, y_train
        print(" Tuning on FULL training set (below cap)")

    # Param grid (kept small for speed)
    param_grid = {
        "n_estimators": [200],
        "max_depth": [4, 6],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0, 3.0],
    }

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        # You can later add scale_pos_weight to handle imbalance
    }

    best_score = -1.0
    best_params = None

    print(" Hyperparameter tuning (binary large-move)...")
    for param_dict in ParameterGrid(param_grid):
        xgb_params = {**base_params, **param_dict}
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train_tune,
            y_train_tune,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        val_f1 = f1_score(y_val, val_pred)

        print(
            f" est={param_dict['n_estimators']:<3} "
            f"depth={param_dict['max_depth']:<2} "
            f"lr={param_dict['learning_rate']:<4} "
            f"sub={param_dict['subsample']:<3} "
            f"col={param_dict['colsample_bytree']:<3} "
            f"lam={param_dict['reg_lambda']:<3} "
            f"F1={val_f1:.4f}"
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_params = xgb_params.copy()

    if best_params is None:
        raise RuntimeError(
            "No best_params found for large-move tuning; check data and grid."
        )

    print(f"\n Best params (large-move): {best_params}")
    print(f" Training final large-move model on FULL {len(X_train):,} samples...")

    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    train_prob = final_model.predict_proba(X_train)[:, 1]
    val_prob = final_model.predict_proba(X_val)[:, 1]
    test_prob = final_model.predict_proba(X_test)[:, 1]

    train_pred = (train_prob >= 0.5).astype(int)
    val_pred = (val_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    model_name = "xgb_large"

    train_metrics = evaluate_binary_split(
        y_train, train_pred, train_prob, "train", config, model_name
    )
    val_metrics = evaluate_binary_split(
        y_val, val_pred, val_prob, "val", config, model_name
    )
    test_metrics = evaluate_binary_split(
        y_test, test_pred, test_prob, "test", config, model_name
    )

    all_metrics = {
        "model_type": "XGB_LargeMove",
        "full_train_size": len(X_train),
        "max_tune_samples": max_tune_samples,
        "best_params": best_params,
        "best_val_f1": float(best_score),
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_model, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test F1 (large-move): {test_metrics['test_f1']:.4f}")
    print(f" Test AUC (large-move): {test_metrics['test_auc']:.4f}")
    return final_model, all_metrics


# ---------- Batch training over KEY_CONFIGS ----------

def batch_train_large_move():
    configs = get_key_configs()
    results_summary = []

    print(f"\n{'=' * 80}")
    print(" BATCH TRAINING XGB LARGE MOVE - KEY CONFIGS")
    print(f"{'=' * 80}")

    for i, config in enumerate(configs, 1):
        try:
            print(f"\n[{i:2d}/4] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_xgb_large_move(config)

            summary = {
                "timeframe": config.timeframe,
                "horizon": config.horizon,
                "n_samples": (
                    metrics["test_n_samples"]
                    + metrics["val_n_samples"]
                    + metrics["train_n_samples"]
                ),
                "test_f1": metrics["test_f1"],
                "test_acc": metrics["test_accuracy"],
                "test_auc": metrics["test_auc"],
                "val_f1": metrics["val_f1"],
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
                    "test_f1": np.nan,
                }
            )

    summary_df = pd.DataFrame(results_summary)
    summary_csv = "models/metrics/xgb_large_move_key_configs_summary.csv"
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print("\n LARGE MOVE SUMMARY TABLE:")
    print(
        summary_df.round(4)
        .sort_values("test_f1", ascending=False, na_position="last")
    )
    print(f"\n Master large-move summary: {summary_csv}")

    return results_summary


# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or "--batch" in sys.argv:
        batch_train_large_move()
    elif "--help" in sys.argv:
        print(
            """
Usage:
  python -m src.models.xgb_large_move          # Batch key configs
  python -m src.models.xgb_large_move 15m 15m  # Single TF/horizon
  python -m src.models.xgb_large_move --help   # This help
"""
        )
    else:
        timeframe = sys.argv[1]
        horizon = sys.argv[2] if len(sys.argv) > 2 else "15m"
        cfg = TrainingConfig(timeframe=timeframe, horizon=horizon)
        model, metrics = train_xgb_large_move(cfg)
        print("\n XGB LARGE MOVE COMPLETE!")