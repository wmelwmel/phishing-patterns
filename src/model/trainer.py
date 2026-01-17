import torch
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from setfit import SetFitModel, Trainer, TrainingArguments

from src.model.data_handler import DatasetHandler
from src.model.heads import build_head
from src.model.hp_optimizer import HyperparameterOptimizer
from src.model.metrics import compute_metrics


class ModelTrainer:
    def __init__(self, model_settings: DictConfig, training_args: DictConfig) -> None:
        self.model_settings = model_settings
        self.training_args = training_args
        self.encoder_name_default = model_settings.encoder_name_default
        self.target_strategy = model_settings.get("target_strategy", "multi-output")

        logger.info(f"Default encoder name: {self.encoder_name_default}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset | None = None,
        head_args: DictConfig | None = None,
        finetune_args: DictConfig | None = None,
    ) -> tuple[Trainer, SetFitModel]:
        model = self._build_model(head_args)
        return self._train_model(model, train_dataset, test_dataset, finetune_args)

    def hyperparameter_search(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset | None,
        golden_dataset: Dataset | None,
        n_trials: int,
        main_run_name: str,
        dataset_handler: DatasetHandler | None = None,
        valid_features: list[int] | None = None,
    ) -> tuple[Trainer, SetFitModel]:
        optimizer = HyperparameterOptimizer(self)
        return optimizer.run_optuna(
            train_dataset,
            test_dataset,
            golden_dataset,
            n_trials,
            main_run_name,
            dataset_handler,
            valid_features,
        )

    def _build_model(self, head_args: DictConfig | None = None, override_model_name: str | None = None) -> SetFitModel:
        model_name = override_model_name or self.encoder_name_default
        model = SetFitModel.from_pretrained(
            model_name,
            multi_target_strategy=self.target_strategy,
            trust_remote_code=True,
        ).to(self.device)

        head_cfg = head_args or self.training_args.get("head_args", {})
        if not head_cfg:
            logger.warning("No head_args provided - using default head parameters.")

        model.model_head = build_head(head_cfg)
        return model

    def _train_model(
        self,
        model: SetFitModel,
        train_dataset: Dataset,
        test_dataset: Dataset | None = None,
        finetune_args: DictConfig | None = None,
    ) -> tuple[Trainer, SetFitModel]:
        encoder_args = finetune_args or self.training_args.get("finetune_args", {})
        if not encoder_args:
            logger.warning("No finetune_args provided - using defaults from TrainingArguments.")

        args = TrainingArguments(**encoder_args)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            metric=compute_metrics,
        )
        logger.info("Starting model training...")
        trainer.train()
        logger.success("Model training finished.")
        return trainer, model
