from src.models.train_utils import TrainingConfig, get_model_paths

cfg = TrainingConfig(timeframe="1h", horizon="1h")
clf_path, metrics_path = get_model_paths(cfg, "xgb")
print("clf_path:", clf_path)
print("metrics_path:", metrics_path)