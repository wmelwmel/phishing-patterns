import mlflow
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from src.logger import setup_logger
from src.mlflow_helper import MLFlowHelper as ml_helper
from src.model.data_handler import DatasetHandler
from src.model.evaluator import ModelEvaluator
from src.model.trainer import ModelTrainer
from src.paths import configs_dir, synthetic_df_path
from src.settings import MLflowSettings
from src.utils import get_config


def main() -> None:
    log_file = setup_logger()
    load_dotenv()
    mlflow_settings = MLflowSettings()
    model_config = get_config("model_config.yaml")

    data_cfg = model_config.data_handler
    train_cfg = model_config.training_args.finetune_args
    model_settings = model_config.model_settings

    hp_prefix = ""
    mlflow_experiment = mlflow_settings.experiment

    if model_settings.hyperparameter_search:
        mlflow_experiment += "_optuna"
        hp_prefix = f"{model_settings.hp_mode}_{model_settings.hp_trials}trials_"

    mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    run_name = ml_helper.get_next_mlflow_run_name(
        f"{hp_prefix}{model_config.training_args.head_args.type}_"
        f"{data_cfg.base_run_name}_{synthetic_df_path.stem}_{data_cfg.target_trigger_count}_"
        f"{train_cfg.num_iterations}_{data_cfg.balancer_mode}_"
        f"{train_cfg.num_epochs}_{train_cfg.batch_size}",
        mlflow_settings.experiment,
    )

    with mlflow.start_run(run_name=run_name):
        try:
            # 1. Get datasets
            logger.info("Preparing datasets...")
            ml_helper.log_config_params(model_config, exclude_keys="training_args")

            dataset_handler = DatasetHandler(data_cfg)
            ds_results = dataset_handler.get_datasets()

            train_dataset, test_dataset, golden_dataset = (
                ds_results.train_dataset,
                ds_results.test_dataset,
                ds_results.golden_dataset,
            )
            valid_features = ds_results.valid_features

            datasets_dict = {
                "train_ds": train_dataset,
                "test_ds": test_dataset,
                "golden_ds": golden_dataset,
                "full_test_ds": ds_results.full_test_dataset,
            }

            ml_helper.log_dataset(datasets_dict)
            mlflow.log_figure(ds_results.matrix_fig, "figures/correlation_matrix.html")
            mlflow.log_param("features_n", len(valid_features))
            ml_helper.save_dir(str(configs_dir), "configs")
            ml_helper.log_prompts()

            logger.success("Datasets prepared successfully")

            # 2. Training process
            logger.info("Starting training...")
            model_trainer = ModelTrainer(model_settings, model_config.training_args)

            if model_settings.hyperparameter_search:
                _, model = model_trainer.hyperparameter_search(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    golden_dataset=golden_dataset,
                    n_trials=model_settings.hp_trials,
                    main_run_name=run_name,
                    dataset_handler=dataset_handler,
                    valid_features=valid_features,
                )
            else:
                _, model = model_trainer.train(train_dataset)

            ml_helper.log_model(model)
            ml_helper.log_model_params(model)

            # 3. Evaluate results
            logger.info("Evaluating results...")
            model_evaluator = ModelEvaluator(model, dataset_handler.features, valid_features)

            datasets_dict.pop("train_ds")
            total_metrics, thresholds = model_evaluator.evaluate_results(datasets_dict)

            if thresholds is not None and len(thresholds) > 0:
                mlflow.log_param("opt_thresholds", np.round(thresholds, 3).tolist())
            ml_helper.log_metrics_or_table(total_metrics)
            logger.success("Training pipeline completed successfully.")
        except Exception as error:
            logger.exception(f"Training pipeline failed: {error}")
            raise
        finally:
            mlflow.log_artifact(str(log_file))


if __name__ == "__main__":
    main()
