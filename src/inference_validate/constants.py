from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES_SUBPATH = "configs/features.yaml"
MODEL_SUBPATH = "setfit_model"
MODELS_LOCAL_DIR = DATA_DIR / "mlflow_models"
