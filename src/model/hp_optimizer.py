import copy
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mlflow
import optuna
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig, ListConfig
from optuna.visualization.matplotlib import plot_param_importances
from setfit import SetFitModel, Trainer

from src.mlflow_helper import MLFlowHelper as ml_helper
from src.model.data_handler import DatasetHandler
from src.model.evaluator import ModelEvaluator
from src.utils import get_config

if TYPE_CHECKING:
    from src.model.trainer import ModelTrainer


class HyperparameterOptimizer:
    def __init__(self, trainer: "ModelTrainer") -> None:
        self.trainer = trainer
        self.hp_config = get_config("hp_config.yaml")
        self.hp_mode = trainer.model_settings.get("hp_mode", "both")  # "head" | "encoder" | "both"
        self._validate_hp_config()

    def run_optuna(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset | None,
        golden_dataset: Dataset | None,
        n_trials: int,
        main_run_name: str,
        dataset_handler: DatasetHandler,
        valid_features: list[int],
        seed: int = 42,
    ) -> tuple[Trainer, SetFitModel]:
        head_args_base = self.trainer.training_args.get("head_args", {})
        head_type = head_args_base.get("type", "catboost").lower()

        logger.info(f"Starting Optuna optimization for head='{head_type}' with {n_trials} trials")

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(
            lambda trial: self._objective(
                trial,
                head_type,
                head_args_base,
                train_dataset,
                test_dataset,
                golden_dataset,
                dataset_handler,
                valid_features,
                main_run_name,
            ),
            n_trials=n_trials,
        )

        logger.success(
            f"Optuna optimization finished. Best F1 = {study.best_value:.4f} (trial №{study.best_trial.number})"
        )
        logger.info(f"Best parameters: {study.best_trial.params}")

        try:
            ax = plot_param_importances(study)
            if ax is not None:
                fig = ax.get_figure()
                if fig is not None:
                    if hasattr(fig, "tight_layout"):
                        fig.tight_layout()
                    mlflow.log_figure(fig, "figures/param_importances.png")
                    plt.close(fig)  # type: ignore[arg-type]
                else:
                    logger.warning("Figure is None, cannot save or close")
            else:
                logger.warning("Axes is None, cannot get figure")
        except Exception as e:
            logger.warning(f"Error while plotting importances: {e}")

        best_params_encoder, best_params_head = self._split_best_params(study.best_trial.params, head_type)
        best_params_encoder.pop("model_name", None)
        finetune_args = copy.deepcopy(self.trainer.training_args.get("finetune_args", {}))
        finetune_args.update(best_params_encoder)
        head_args = self._prepare_head_args(head_args_base, head_type, best_params_head)

        best_model = self.trainer._build_model(head_args)
        trainer, model = self.trainer._train_model(best_model, train_dataset, test_dataset, finetune_args)

        logger.success("Best model retrained successfully with optimal hyperparameters.")
        return trainer, model

    def _objective(
        self,
        trial: optuna.Trial,
        head_type: str,
        head_args_base: dict,
        train_dataset: Dataset,
        test_dataset: Dataset | None,
        golden_dataset: Dataset | None,
        dataset_handler: DatasetHandler,
        valid_features: list[int],
        main_run_name: str,
    ) -> float:
        params_encoder, params_head = self._suggest_params(trial, head_type)
        encoder_name = params_encoder.pop("model_name", None)
        encoder_label = encoder_name or "default_encoder"

        run_name = f"{main_run_name}_trial_{trial.number}_{encoder_label}"

        with mlflow.start_run(nested=True, run_name=run_name):
            if encoder_name:
                logger.info(f"Using encoder '{encoder_name}' for trial {trial.number}")

            finetune_args = copy.deepcopy(self.trainer.training_args.get("finetune_args", {}))
            finetune_args.update(params_encoder)

            head_args = self._prepare_head_args(head_args_base, head_type, params_head)

            model = self.trainer._build_model(head_args, encoder_name)
            _, _ = self.trainer._train_model(model, train_dataset, test_dataset, finetune_args)

            evaluator = ModelEvaluator(model, dataset_handler.features, valid_features)
            datasets_dict = {"test_ds": test_dataset}
            if golden_dataset is not None:
                datasets_dict["golden_ds"] = golden_dataset

            metrics = evaluator.evaluate_results(datasets_dict)

            ml_helper.log_metrics_or_table(metrics)
            ml_helper.log_model_params(model)

            score = metrics.get("test_ds_f1_w")
            if score is None:
                raise ValueError("Metric 'test_ds_f1_w' not found in evaluation results. Check metric configuration.")
            return float(score)

    def _suggest_params(self, trial: optuna.Trial, head_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
        params_encoder, params_head = {}, {}

        if self.hp_mode in ["encoder", "both"]:
            for k, cfg in self.hp_config.get("encoder", {}).items():
                params_encoder[k] = self._suggest(trial, k, cfg)

        if self.hp_mode in ["head", "both"]:
            if head_type == "logreg":
                params_head = self._suggest_logreg_params(trial, self.hp_config["head"][head_type])
            else:
                for k, cfg in self.hp_config.get("head", {}).get(head_type, {}).items():
                    params_head[k] = self._suggest(trial, k, cfg)

        return params_encoder, params_head

    def _suggest(self, trial: optuna.Trial, name: str, cfg: dict) -> bool | str | float | None:
        if "values" in cfg:
            return trial.suggest_categorical(name, cfg["values"])

        low, high = cfg["low"], cfg["high"]
        step, log = cfg.get("step"), cfg.get("log", False)

        is_int = all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in [low, high])
        if is_int:
            return trial.suggest_int(name, int(low), int(high), step=int(step) if step else 1)
        else:
            return trial.suggest_float(name, float(low), float(high), step=step, log=log)

    def _suggest_logreg_params(self, trial: optuna.Trial, hp_cfg: dict) -> dict[str, Any]:
        params = {}

        solver = trial.suggest_categorical("solver", hp_cfg["solver"]["values"])
        params["solver"] = solver

        allowed_penalties = hp_cfg["penalty"][solver]
        penalty_index = trial.suggest_int("penalty_index", 0, len(allowed_penalties) - 1)
        params["penalty"] = allowed_penalties[penalty_index]

        for k, cfg in hp_cfg.items():
            if k in ["solver", "penalty"]:
                continue
            params[k] = self._suggest(trial, k, cfg)

        return params

    def _prepare_head_args(
        self, base_args: dict[str, Any], head_type: str, params_head: dict[str, Any]
    ) -> dict[str, Any]:
        new_args = copy.deepcopy(base_args)
        new_args[head_type] = {**(new_args.get(head_type) or {}), **params_head}
        return new_args

    def _split_best_params(self, best_params: dict[str, Any], head_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
        encoder_keys = self.hp_config.get("encoder", {}).keys()
        head_keys = self.hp_config.get("head", {}).get(head_type, {}).keys()

        params_encoder = {k: v for k, v in best_params.items() if k in encoder_keys}
        params_head = {k: v for k, v in best_params.items() if k in head_keys}

        return params_encoder, params_head

    def _validate_hp_config(self) -> None:
        encoder_cfg = self.hp_config.get("encoder", {})
        if not isinstance(encoder_cfg, dict | DictConfig):
            raise ValueError("encoder section in hp_config must be a dict")
        for name, cfg in encoder_cfg.items():
            self._validate_param_cfg(cfg, f"encoder.{str(name)}")

        head_cfg = self.hp_config.get("head", {})
        if not isinstance(head_cfg, dict | DictConfig):
            raise ValueError("head section in hp_config must be a dict")
        for head_type_key, head_params in head_cfg.items():
            if not isinstance(head_params, dict | DictConfig):
                raise ValueError(f"head.{str(head_type_key)} in hp_config must be a dict")
            if head_type_key == "logreg":
                solver_is_defined = "solver" in head_params
                penalty_is_defined = "penalty" in head_params

                solver_cfg = head_params.get("solver")
                penalty_cfg = head_params.get("penalty")

                if solver_is_defined:
                    if solver_cfg is None:
                        raise ValueError("head.logreg.solver is defined but empty (None/null)")
                    if not isinstance(solver_cfg, (dict, DictConfig)):
                        raise ValueError(f"head.logreg.solver must be a dict, got {type(solver_cfg)}")
                    if "values" not in solver_cfg:
                        raise ValueError("head.logreg.solver must contain 'values'")
                    solver_values = solver_cfg["values"]
                    if not isinstance(solver_values, (list, ListConfig)) or len(solver_values) == 0:
                        raise ValueError("head.logreg.solver.values must be a non-empty list")
                else:
                    solver_values = None

                if penalty_is_defined:
                    if penalty_cfg is None:
                        raise ValueError("head.logreg.penalty is defined but empty (None/null)")
                    if not isinstance(penalty_cfg, (dict, DictConfig)):
                        raise ValueError("head.logreg.penalty must be a dict")
                    if len(penalty_cfg) == 0:
                        raise ValueError("head.logreg.penalty must not be empty")

                    for solver_name, penalties in penalty_cfg.items():
                        if not isinstance(penalties, (list, ListConfig)):
                            raise ValueError(f"head.logreg.penalty.{str(solver_name)} must be a list")
                        if len(penalties) == 0:
                            raise ValueError(f"head.logreg.penalty.{str(solver_name)} cannot be empty")

                    if solver_values is not None:
                        penalty_solvers = list(penalty_cfg.keys())
                        missing = set(solver_values) - set(penalty_solvers)
                        extra = set(penalty_solvers) - set(solver_values)
                        if missing:
                            raise ValueError(f"penalty missing solvers from solver.values: {missing}")
                        if extra:
                            logger.warning(f"penalty defines solvers not listed in solver.values: {extra}")
                else:
                    if solver_is_defined:
                        logger.warning("head.logreg.penalty not defined (optional)")
                    else:
                        logger.warning("head.logreg.solver and penalty not defined (optional)")
                for k, cfg in head_params.items():
                    if k in ("solver", "penalty"):
                        continue
                    self._validate_param_cfg(cfg, f"head.{str(head_type_key)}.{str(k)}")
            else:
                for name_key, cfg in head_params.items():
                    self._validate_param_cfg(cfg, f"head.{str(head_type_key)}.{str(name_key)}")

        logger.success("Hyperparameter configuration validated successfully.")

    def _validate_param_cfg(self, cfg: dict | None, name: str) -> None:
        if cfg is None:
            raise ValueError(f"{name}: hyperparameter config is None, got: '{name}: {cfg}'")

        if "values" in cfg:
            vals = cfg.get("values")
            if not isinstance(vals, list | ListConfig) or len(vals) == 0:
                raise ValueError(f"{name}: 'values' must be a non-empty list in hp_config, got: {cfg}")
            return

        if "low" not in cfg or "high" not in cfg:
            raise ValueError(f"Hyperparameter '{name}' must have 'low' and 'high' defined, got: {cfg}")

        low, high = cfg["low"], cfg["high"]

        low, high = cfg["low"], cfg["high"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(
                f"{name}: 'low' and 'high' must be numeric, got: low={low} ({type(low)}), high={high} ({type(high)})"
            )

        if low >= high:
            raise ValueError(f"{name}: 'low' must be less than 'high' (got {low} >= {high})")
