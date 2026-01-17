from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseS3Settings(BaseSettings):
    """Base S3 settings with common fields."""

    s3_url: str
    s3_bucket: str
    s3_prefix: str
    s3_key_id: str
    s3_secret: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class S3SettingsMy(BaseS3Settings):
    """Settings for MY DB S3."""

    model_config = SettingsConfigDict(env_prefix="MY_DB_")


class S3SettingsYandex(BaseS3Settings):
    """Settings for MY Yandex S3."""

    model_config = SettingsConfigDict(env_prefix="MY_YANDEX_DB_")


class ArchiveExtractorSettings(BaseSettings):
    """Settings for archive extractor."""

    infected_pass: str = Field(..., description="Password for pass_infected.7z")
    mails_pass: str = Field(..., description="Password for mails_eml.zip")

    model_config = SettingsConfigDict(env_prefix="ARCHIVE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class LLMSettings(BaseSettings):
    """Settings class for LLM."""

    api_key: str
    base_url: str

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class MLflowSettings(BaseSettings):
    """Settings class for MLflow."""

    tracking_uri: str
    tracking_username: str
    tracking_password: str
    experiment: str

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
