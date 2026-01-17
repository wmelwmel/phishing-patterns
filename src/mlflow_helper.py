import copy
import re
import tempfile
from pathlib import Path
from typing import Union

import mlflow
import pandas as pd
from datasets import Dataset
from loguru import logger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from setfit import SetFitModel

from src.paths import jinja_templates_dir, mlflow_models_local_dir
from src.settings import MLflowSettings


class MLFlowHelper:
    @staticmethod
    def get_next_mlflow_run_name(base_run_name: str, mlflow_experiment: str) -> str:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment)

        if experiment is None:
            raise ValueError(f"Experiment {mlflow_experiment} not found")

        max_index = 0
        runs_df: pd.DataFrame = mlflow.search_runs([experiment.experiment_id])

        if runs_df.empty:
            return f"{base_run_name}_{max_index}"
        else:
            existing_run_names = runs_df["tags.mlflow.runName"].tolist()

            pattern = re.compile(rf"^{re.escape(base_run_name)}(_\d+)?$")

            for name in existing_run_names:
                match = pattern.match(name)
                if match:
                    if match.group(1):
                        current_index = int(match.group(1)[1:])
                        max_index = max(max_index, current_index)

            next_index = max_index + 1
            return f"{base_run_name}_{next_index}"

    @staticmethod
    def save_dir(dir_path: str, artifact_path: str = "default") -> None:
        mlflow.log_artifacts(dir_path, artifact_path=artifact_path)

    @staticmethod
    def log_metrics_or_table(
        metrics: Union[dict, pd.DataFrame], mlflow_table: bool = False, artifact_file: str = "data.json"
    ) -> None:
        if isinstance(metrics, dict):
            rounded_metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
            mlflow.log_metrics(rounded_metrics)
        elif isinstance(metrics, pd.DataFrame) and mlflow_table:
            mlflow.log_table(
                data=metrics.reset_index().to_dict(orient="list"), artifact_file=f"datasets/{artifact_file}"
            )

    @staticmethod
    def log_prompts(art_dir: str = "prompts") -> None:
        for prompt_path in jinja_templates_dir.rglob("*.j2"):
            prompt_name = prompt_path.stem
            with open(prompt_path, "r") as f:
                j2_content = f.read()
            mlflow.log_text(j2_content, f"{art_dir}/{prompt_name}.txt")

    @staticmethod
    def register_prompt(
        template_filename: str, experiment: str, run_name: str, prompt_name: str = "phishing_labelling_prompt"
    ) -> tuple[str, str]:
        prompt_path = jinja_templates_dir / template_filename
        prompt_content = prompt_path.read_text(encoding="utf-8")
        prompt = mlflow.register_prompt(
            name=prompt_name,
            template=prompt_content,
            tags={
                "experiment": experiment,
                "run_name": run_name,
            },
        )
        logger.info(f"Created prompt '{prompt.name}' (version {prompt.version})")
        return prompt.name, prompt.version

    @staticmethod
    def log_dataset(datasets_info: dict[str, Dataset | None]) -> None:
        for context, dataset in datasets_info.items():
            if isinstance(dataset, Dataset):
                df = dataset.to_pandas()
                table_path = f"datasets/{context}.json"
                mlflow.log_table(df, table_path)
                mlflow_dataset = mlflow.data.from_pandas(df, source=table_path)  # type: ignore[attr-defined]
                mlflow.log_input(mlflow_dataset, context=context)
                logger.info(f"Dataset {context} saved successfully")
            else:
                logger.warning(f"Saving error. Invalid format of {context} dataset")

    @staticmethod
    def log_model(model: SetFitModel, artifact_path: str = "setfit_model") -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            mlflow.log_artifacts(tmp_dir, artifact_path=artifact_path)

    @staticmethod
    def log_config_params(config: DictConfig, exclude_keys: list[str] | str | None = None) -> None:
        config_to_log = copy.deepcopy(config)
        exclude_keys = [exclude_keys] if isinstance(exclude_keys, str) else (exclude_keys or [])

        for key in exclude_keys:
            config_to_log.pop(key, None)

        for key, value in config_to_log.items():  # type: ignore[assignment]
            if isinstance(value, DictConfig):
                MLFlowHelper.log_config_params(value)
            else:
                mlflow.log_param(key, value)  # type: ignore[arg-type]

    @staticmethod
    def log_model_params(model: SetFitModel) -> None:
        mlflow.log_params(model.model_head.get_params())
        mlflow.log_param("body_encoder", model.model_card_data.st_id)

    @staticmethod
    def download_mlflow_model(run_id: str, model_dirname: str) -> tuple[Path, Path]:
        mlflow.set_tracking_uri(MLflowSettings().tracking_uri)

        mlflow_models_local_dir.mkdir(exist_ok=True)
        client = MlflowClient()
        run_id_model_dir = mlflow_models_local_dir / run_id
        model_path = run_id_model_dir / model_dirname

        if model_path.exists():
            logger.info(f"Model already downloaded ({model_path}).")
        else:
            logger.info(f"Downloading model from MLflow to {model_path}...")
            client.download_artifacts(run_id, path=model_dirname, dst_path=str(run_id_model_dir))
            logger.info("Download completed.")

        return model_path, run_id_model_dir
