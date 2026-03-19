# src/models/train_random_forest.py

import argparse
import numpy as np  # Make sure numpy is imported
from sklearn.ensemble import RandomForestClassifier

from train_utils import (
    TrainingConfig, 
    load_dataset, 
    make_timeseries_splits, 
    evaluate_classification, 
    save_model,
    save_metrics,
    get_model_paths
)

print("--- Training Random Forest Model ---")

# 1. CONFIGURE your training run
config = TrainingConfig(timeframe="1h", horizon="4h")

# 2. LOAD the 3D data using the utility
X, y, t, _ = load_dataset(config)

# =================================================================
# THE FIX: FLATTEN THE 3D DATA TO 2D FOR THE RANDOM FOREST
# =================================================================
print(f"Original X shape (3D): {X.shape}")
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape((n_samples, n_timesteps * n_features))
print(f"Reshaped X shape (2D): {X_reshaped.shape}")
# =================================================================

# 3. SPLIT the data into train, validation, and test sets
train_idx, val_idx, test_idx = make_timeseries_splits(t, config)
# Use the RESHAPED data for splitting!
X_train, y_train = X_reshaped[train_idx], y[train_idx]
X_val, y_val = X_reshaped[val_idx], y[val_idx]
X_test, y_test = X_reshaped[test_idx], y[test_idx]

# 4. DEFINE your Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=config.random_state,
    n_jobs=-1
)

# 5. TRAIN the model
print(f"Training Random Forest on {len(X_train)} samples...")
rf_model.fit(X_train, y_train)
print("Training complete!")

# 6. EVALUATE the model on the validation set
print("Evaluating on validation set...")
y_val_pred = rf_model.predict(X_val)
val_metrics = evaluate_classification(y_val, y_val_pred, "val")

# 7. SAVE your trained model and its metrics
model_path, metrics_path = get_model_paths(config, "random_forest")
save_model(rf_model, model_path)
save_metrics(val_metrics, metrics_path)

print(f"\nRandom Forest model and metrics saved successfully!")
