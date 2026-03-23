"""
XGBoost baseline + evaluation for CryptoPulse AI.
- Multi-class classification on flattened sliding windows (48x21 → 1008 features).
- Label remap [-2,-1,0,1,2] → [0..4] for XGBoost.
- Hyperparameter tuning on capped subsample + full-data final fit with early stopping.
- Adds confusion matrices, per-class metrics, and prediction CSV export.
- Restricted to key configs: 15m/15m, 15m/1h, 1h/1h, 1h/4h.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from src.models.train_utils import (
    TrainingConfig,
    load_dataset,
    make_timeseries_splits,
    evaluate_classification,  # kept for backward-compat if needed
    save_model,
    save_metrics,
    get_model_paths,
)

# GLOBAL LABEL MAPPING: CryptoPulse 5-class -> XGBoost [0-4]
LABEL_MAP = {
    -2: 0,
    -1: 1,
    0: 2,
    1: 3,
    2: 4,
}
NEUTRAL_CLASS = 2  # Maps to original 0 (neutral)

# Human-readable names for mapped classes 0..4
CLASS_NAMES = {
    0: "Bearish",
    1: "SideBear",
    2: "Neutral",
    3: "SideBull",
    4: "Bullish",
}

KEY_CONFIGS = [
    ("15m", "15m"),
    ("15m", "1h"),
    ("1h", "1h"),
    ("1h", "4h"),
    ("4h", "4h"),
    ("4h", "1d"),
]

# ---------- Core helpers ----------

def flatten_features(X: np.ndarray) -> np.ndarray:
    """Flatten sliding windows: (batch, window=48, features=21) → (batch, 1008)."""
    return X.reshape(X.shape[0], -1)


def remap_labels(y: np.ndarray) -> np.ndarray:
    """Remap CryptoPulse labels [-2,-1,0,1,2] → XGBoost [0,1,2,3,4]. Handle invalids."""
    y_mapped = np.vectorize(LABEL_MAP.get)(y)
    # Replace any unmapped/NaN → neutral (2)
    y_mapped = np.where(np.isnan(y_mapped), NEUTRAL_CLASS, y_mapped).astype(np.int32)
    return y_mapped


def get_key_configs():
    """Return TrainingConfig objects for the selected timeframe/horizon pairs."""
    return [TrainingConfig(timeframe=tf, horizon=hz) for tf, hz in KEY_CONFIGS]


# ---------- Evaluation artifacts (Option A) ----------

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ✓ Confusion matrix saved: {save_path}")


def save_predictions_and_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    config: TrainingConfig,
    model_name: str,
    metrics: dict,
) -> str:
    """Save predictions CSV + per-class metrics into metrics dict."""
    # Predictions DataFrame
    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": (y_true == y_pred).astype(int),
            "class_name_true": [CLASS_NAMES[int(i)] for i in y_true],
            "class_name_pred": [CLASS_NAMES[int(i)] for i in y_pred],
        }
    )

    pred_csv = (
        f"models/predictions/"
        f"{config.timeframe}_{config.horizon}_{model_name}_{split_name}_predictions.csv"
    )
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    pred_df.to_csv(pred_csv, index=False)

    # Detailed classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CLASS_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )

    # Per-class metrics into all_metrics
    for class_name in CLASS_NAMES.values():
        key_base = class_name.lower().replace(" ", "_")
        cls_stats = report.get(class_name, {})
        metrics[f"{split_name}_f1_{key_base}"] = cls_stats.get("f1-score", 0.0)
        metrics[f"{split_name}_precision_{key_base}"] = cls_stats.get("precision", 0.0)
        metrics[f"{split_name}_recall_{key_base}"] = cls_stats.get("recall", 0.0)

    print(f" ✓ Predictions saved: {pred_csv} ({len(pred_df)} samples)")
    return pred_csv


def enhanced_evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    config: TrainingConfig,
    model_name: str,
) -> dict:
    """Enhanced evaluation: macro metrics + confusion matrix + predictions CSV."""
    metrics = {
        f"{split_name}_n_samples": int(len(y_true)),
        f"{split_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{split_name}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    # Save predictions + per-class stats
    pred_csv = save_predictions_and_metrics(
        y_true, y_pred, split_name, config, model_name, metrics
    )

    # Confusion matrix PNG
    cm_path = (
        f"models/metrics/"
        f"{config.timeframe}_{config.horizon}_{model_name}_{split_name}_cm.png"
    )
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plot_confusion_matrix(
        y_true,
        y_pred,
        list(CLASS_NAMES.values()),
        cm_path,
        title=f"CM: {config.timeframe}/{config.horizon} {split_name}",
    )

    metrics[f"{split_name}_pred_csv"] = pred_csv
    metrics[f"{split_name}_cm_path"] = cm_path
    metrics["class_names"] = str(CLASS_NAMES)
    return metrics


# ---------- Main XGBoost training ----------

def train_xgboost(
    config: TrainingConfig,
    max_tune_samples: int = 20_000,
    enhanced_eval: bool = True,
) -> tuple:
    """
    Train XGBoost with:
    - Label remapping [-2..2] → [0..4] for XGBoost compatibility
    - Hyperparameter tuning on a capped subset of the training data
    - Final model refit on FULL training data with best hyperparameters + early stopping
    - Optional enhanced evaluation artifacts (confusion matrix, per-class metrics, CSVs)
    """
    print(f"\n{'=' * 60}")
    print(f" XGBoost: {config.timeframe}/{config.horizon}")
    print(f"{'=' * 60}")

    # 1. Load & prepare
    X, y, t, label_cols = load_dataset(config)
    print(f" Original classes: {np.unique(y)}")

    X_flat = flatten_features(X)
    y_mapped = remap_labels(y)
    print(f" Remapped classes: {np.unique(y_mapped)}")

    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)

    X_train, y_train = X_flat[train_idx], y_mapped[train_idx]
    X_val, y_val = X_flat[val_idx], y_mapped[val_idx]
    X_test, y_test = X_flat[test_idx], y_mapped[test_idx]

    print(
        f" Full data: X_train({len(X_train):,}) | "
        f"X_flat({X_flat.shape[1]})"
    )

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

    # 3. Small param grid for XGBoost
    param_grid = {
        "n_estimators": [200],
        "max_depth": [4, 6],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0, 3.0],
    }

    base_params = {
        "objective": "multi:softprob",
        "num_class": 5,  # Matches remapped [0,1,2,3,4]
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }

    best_score = -1.0
    best_params = None

    print(" Hyperparameter tuning...")
    for param_dict in ParameterGrid(param_grid):
        xgb_params = {**base_params, **param_dict}
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train_tune,
            y_train_tune,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")

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
            "No best_params found during XGBoost tuning; check data and grid."
        )

    print(f"\n Best params found: {best_params}")
    print(f" Training final XGBoost on FULL {len(X_train):,} samples...")

    # 4. Final model with early stopping on validation
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 5. Evaluate final model
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    test_pred = final_model.predict(X_test)

    model_name = "xgb"

    if enhanced_eval:
        train_metrics = enhanced_evaluate_classification(
            y_train, train_pred, "train", config, model_name
        )
        val_metrics = enhanced_evaluate_classification(
            y_val, val_pred, "val", config, model_name
        )
        test_metrics = enhanced_evaluate_classification(
            y_test, test_pred, "test", config, model_name
        )
    else:
        # Fallback to existing simple metrics helper if you ever want it
        train_metrics = evaluate_classification(y_train, train_pred, "train")
        val_metrics = evaluate_classification(y_val, val_pred, "val")
        test_metrics = evaluate_classification(y_test, test_pred, "test")

    all_metrics = {
        "model_type": "XGBoost",
        "label_map": str(LABEL_MAP),
        "class_names": str(CLASS_NAMES),
        "full_train_size": len(X_train),
        "max_tune_samples": max_tune_samples,
        "best_params": best_params,
        "best_val_f1": float(best_score),
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    # 6. Save
    model_path, metrics_path = get_model_paths(config, model_name)
    save_model(final_model, model_path)
    save_metrics(all_metrics, metrics_path)

    print(f"\n Test F1: {all_metrics['test_macro_f1']:.4f}")
    print(f" Test predictions: {all_metrics.get('test_pred_csv', 'n/a')}")
    print(f" Test CM: {all_metrics.get('test_cm_path', 'n/a')}")
    return final_model, all_metrics


def batch_train_key_configs():
    """Train XGBoost on the 4 key timeframe/horizon combinations with enhanced eval."""
    configs = get_key_configs()
    results_summary = []

    print(f"\n{'=' * 80}")
    print(" BATCH TRAINING XGBOOST - KEY CONFIGS")
    print(f"{'=' * 80}")

    for i, config in enumerate(configs, 1):
        try:
            print(f"\n[{i:2d}/4] {config.timeframe:>3s}/{config.horizon:>3s}")
            model, metrics = train_xgboost(config, enhanced_eval=True)

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
                "test_pred_csv": metrics.get("test_pred_csv", ""),
                "test_cm_path": metrics.get("test_cm_path", ""),
            }
            results_summary.append(summary)

        except Exception as e:
            print(f" FAILED: {e}")
            results_summary.append(
                {
                    "timeframe": config.timeframe,
                    "horizon": config.horizon,
                    "error": str(e)[:200],
                    "test_f1": np.nan,  # Ensure column exists for sorting
                }
            )

    summary_df = pd.DataFrame(results_summary)
    summary_csv = "models/metrics/xgb_key_configs_summary.csv"
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
        print(
            """
Usage:
  python -m src.models.xgboost_model          # Batch key configs (enhanced eval)
  python -m src.models.xgboost_model 15m 15m  # Single TF/horizon
  python -m src.models.xgboost_model --help   # This help
"""
        )
    else:
        timeframe = sys.argv[1]
        horizon = sys.argv[2] if len(sys.argv) > 2 else "15m"
        cfg = TrainingConfig(timeframe=timeframe, horizon=horizon)
        model, metrics = train_xgboost(cfg, enhanced_eval=True)
        print("\n XGBoost COMPLETE!")