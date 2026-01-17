from pathlib import Path

from config import languages, messages_to_process, run_id, use_gpu, use_txt_split, verbose
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    messages_to_process: Path | list[str] = messages_to_process
    run_id: str = run_id
    languages: list[str] = languages
    use_gpu: bool = use_gpu
    use_txt_split: bool = use_txt_split
    verbose: bool = verbose


class MLflowSettings(BaseSettings):
    """Settings class for MLflow."""

    tracking_uri: str
    tracking_username: str
    tracking_password: str

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
