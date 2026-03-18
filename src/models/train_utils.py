"""
Shared utilities for CryptoPulse AI traditional ML training pipeline.
Supports ALL timeframes: 5m, 15m, 1h, 4h, 1d, 1w with their respective horizons.
Chronological 70/15/15 splits (train/val/test) to prevent look-ahead bias.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]

HORIZONS_BY_TF = {
    "5m": ["5m", "15m"],
    "15m": ["15m", "1h"], 
    "1h": ["1h", "4h"],
    "4h": ["4h", "1d"],
    "1d": ["1d", "1w"],
    "1w": ["1w"]
}

@dataclass
class TrainingConfig:
    """Configuration for a single training run."""
    timeframe: str                    # REQUIRED: "5m", "15m", "1h", "4h", "1d", "1w"
    horizon: str                      # REQUIRED: Matches HORIZONS_BY_TF[timeframe]
    label_column: Optional[str] = None
    train_split: float = 0.7          # 70% train
    val_split: float = 0.15           # 15% validation  
    random_state: int = 42
    use_class_weights: bool = True
    data_dir: str = "data/processed"
    models_dir: str = "models"
    metrics_dir: str = "models/metrics"
    
    # Computed paths (set in __post_init__)
    dataset_prefix: str = field(init=False)
    model_path_template: str = field(init=False)
    metrics_path_template: str = field(init=False)
    
    def __post_init__(self):
        """Validate config and compute paths."""
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}, got '{self.timeframe}'")
        
        if self.horizon not in HORIZONS_BY_TF[self.timeframe]:
            valid = HORIZONS_BY_TF[self.timeframe]
            raise ValueError(f"horizon '{self.horizon}' not valid for {self.timeframe}. Valid: {valid}")
        
        # Compute paths
        self.dataset_prefix = f"btc_{self.timeframe}_featured_labels_"
        self.model_path_template = f"models/{{model_name}}/{self.timeframe}_{self.horizon}.pkl"
        self.metrics_path_template = f"models/metrics/{{model_name}}_{self.timeframe}_{self.horizon}_metrics.json"
        
        # Auto-detect label column
        if self.label_column is None:
            self.label_column = f"label_{self.horizon}"
        
        print(f"Config: {self.timeframe}/{self.horizon} → {self.label_column}")

def load_dataset(config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:

    base_path = Path(config.data_dir)
    
    prefix = f"btc_{config.timeframe}_featured_"

    X_path = base_path / f"{prefix}labels_X.npy"
    y_path = base_path / f"{prefix}labels_y.npy" 
    t_path = base_path / f"{prefix}labels_t.npy"
    label_cols_path = base_path / f"{prefix}label_cols.npy"
    
    print(f"Target files:")
    print(f" X: {X_path}")
    print(f" y: {y_path}")
    print(f" t: {t_path}")
    print(f" label_cols: {label_cols_path}")
    
    # Load ALL 4 files
    X = np.load(X_path)
    y = np.load(y_path)
    t = np.load(t_path)
    label_cols_raw = np.load(label_cols_path)
    
    # Convert label_cols to list of strings (usually bytes → strings)
    if label_cols_raw.dtype.kind in ['S', 'U']:  # String/bytes arrays
        label_cols = label_cols_raw.astype(str).tolist()
    else:
        label_cols = label_cols_raw.tolist()
    
    print(f"Available labels: {label_cols[:3]}...")  # Show first 3
    
    # Select label column
    try:
        label_idx = np.where(np.array(label_cols) == config.label_column)[0][0]
        y_selected = y[:, label_idx].astype(int)
    except (IndexError, ValueError):
        print(f"'{config.label_column}' not in {label_cols}")
        raise ValueError(f"Label '{config.label_column}' not found")
    
    print(f"SUCCESS: X({X.shape}), y({y_selected.shape}), classes: {np.unique(y_selected)}")
    return X, y_selected, t, label_cols


def make_timeseries_splits(t: np.ndarray, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/val/test splits (70/15/15) - NO SHUFFLING.
    Prevents look-ahead bias critical for time-series.
    """
    n_samples = len(t)
    train_end = int(n_samples * config.train_split)
    val_end = int(n_samples * (config.train_split + config.val_split))
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)
    
    print(f"Splits: train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")
    print(f"Range: {pd.to_datetime(t[0])} → {pd.to_datetime(t[-1])}")
    
    return train_idx, val_idx, test_idx

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, set_name: str) -> Dict[str, float]:
    """Multi-class metrics: accuracy, macro F1, confusion matrix."""
    metrics = {
        f"{set_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{set_name}_macro_f1": float(f1_score(y_true, y_pred, average='macro')),
        f"{set_name}_n_samples": int(len(y_true)),
        f"{set_name}_n_classes": int(len(np.unique(y_true)))
    }
    
    # Confusion matrix (flattened for JSON)
    cm = confusion_matrix(y_true, y_pred)
    metrics[f"{set_name}_confusion_matrix"] = cm.flatten().tolist()
    
    print(f"{set_name}: Acc={metrics[f'{set_name}_accuracy']:.3f}, F1={metrics[f'{set_name}_macro_f1']:.3f}")
    return metrics

def save_model(model: Any, model_path: str):
    """Save model with joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model: {model_path}")

def save_metrics(metrics: Dict[str, Any], metrics_path: str):
    """Save metrics as JSON."""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics_path}")

def get_model_paths(config: TrainingConfig, model_name: str) -> Tuple[str, str]:
    """Get consistent paths for model + metrics."""
    model_path = config.model_path_template.format(model_name=model_name)
    metrics_path = config.metrics_path_template.format(model_name=model_name)
    return model_path, metrics_path

# BATCH SUPPORT
def get_all_configs() -> List[TrainingConfig]:
    """Generate ALL valid timeframe/horizon combinations (11 total)."""
    all_configs = []
    for tf in TIMEFRAMES:
        for horizon in HORIZONS_BY_TF[tf]:
            all_configs.append(TrainingConfig(timeframe=tf, horizon=horizon))
    return all_configs

# TESTING
def test_utils(config: TrainingConfig):
    """End-to-end test of all utilities."""
    print(f"\nTesting {config.timeframe}/{config.horizon}...")
    
    # 1. Load
    X, y, t, label_cols = load_dataset(config)
    
    # 2. Split  
    train_idx, val_idx, test_idx = make_timeseries_splits(t, config)
    
    # 3. Dummy predictions for testing
    y_train_pred = np.zeros_like(y[train_idx])
    y_val_pred = np.zeros_like(y[val_idx])
    y_test_pred = np.zeros_like(y[test_idx])
    
    # 4. Evaluate
    train_metrics = evaluate_classification(y[train_idx], y_train_pred, "train")
    val_metrics = evaluate_classification(y[val_idx], y_val_pred, "val")
    test_metrics = evaluate_classification(y[test_idx], y_test_pred, "test")
    
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}
    
    # 5. Save test artifacts
    dummy_model_path, dummy_metrics_path = get_model_paths(config, "dummy")
    save_model("dummy_model_object", dummy_model_path)
    save_metrics(all_metrics, dummy_metrics_path)
    
    print("PASSED")
    return all_metrics

def test_all_timeframes():
    """Test ALL 11 timeframe/horizon combinations."""
    print("Testing ALL timeframes...")
    all_configs = get_all_configs()
    
    success = 0
    for i, config in enumerate(all_configs, 1):
        try:
            print(f"\n[{i:2d}/11] {config.timeframe}/{config.horizon}")
            test_utils(config)
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
    
    print(f"\n RESULT: {success}/11 timeframes successful!")
    return success == 11

# CLI ENTRYPOINT
def main_cli():
    parser = argparse.ArgumentParser(description="CryptoPulse AI Training Utilities")
    parser.add_argument("--timeframe", choices=TIMEFRAMES, help="Single timeframe")
    parser.add_argument("--horizon", help="Horizon for single TF")
    parser.add_argument("--test-all", action="store_true", help="Test all 11 TF/horizon combos")
    parser.add_argument("--list-configs", action="store_true", help="List all valid configs")
    
    args = parser.parse_args()
    
    if args.list_configs:
        configs = get_all_configs()
        print("All valid configs:")
        for i, c in enumerate(configs, 1):
            print(f"  {i:2d}. {c.timeframe:>3s}/{c.horizon:>3s} → {c.label_column}")
        return
    
    if args.test_all:
        test_all_timeframes()
        return
    
    if args.timeframe and args.horizon:
        config = TrainingConfig(timeframe=args.timeframe, horizon=args.horizon)
        test_utils(config)
    else:
        print("Run parameters: --timeframe 1h --horizon 4h   OR   --test-all   OR   --list-configs")
        parser.print_help()

if __name__ == "__main__":
    main_cli()
