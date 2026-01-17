import numpy as np
from datasets import Dataset
from loguru import logger
from setfit import SetFitModel

from src.datamodels import EvalData
from src.model.metrics import compute_all_metrics
from src.model.thresholds_opt import ThresholdOptimizer, ThresholdVisualizer
from src.utils import get_config


class ModelEvaluator:
    def __init__(self, model: SetFitModel, features: list[dict], valid_features: list[int]) -> None:
        self.model = model
        feature_names = [f["name"] for f in features]
        self.valid_feature_names = [feature_names[i] for i in valid_features]

        eval_config = get_config("evaluator_config.yaml")
        self.log_reports = eval_config.log_reports
        self.plot_feature = eval_config.plot_feature

        self.custom_thresholds = eval_config.get("custom_thresholds")
        self.thresholds_ds_name = eval_config.get("thresholds_ds_name", "test_ds")
        self.thresholds_metric = eval_config.get("thresholds_metric", "f1_w")
        self.min_thr = eval_config.get("min_threshold", 0.05)
        self.max_thr = eval_config.get("max_threshold", 0.95)
        self.step_thr = eval_config.get("step_thr", 0.01)

        self.threshold_optimizer = ThresholdOptimizer(
            metric=self.thresholds_metric,
            min_threshold=self.min_thr,
            max_threshold=self.max_thr,
            step=self.step_thr,
        )
        self.visualizer = ThresholdVisualizer(self.valid_feature_names)

    def evaluate_results(
        self,
        evaluation_dict: dict[str, Dataset | None],
    ) -> tuple[dict, np.ndarray | None]:
        total_metrics: dict[str, float | int] = {}

        thr_ds = evaluation_dict.get(self.thresholds_ds_name)
        if thr_ds is None:
            msg = f"Dataset '{self.thresholds_ds_name}' not found for threshold optimization."
            logger.error(msg)
            raise ValueError(msg)

        y_true_thr, y_probs_thr = self._get_probabilities(thr_ds)

        if self.custom_thresholds:
            logger.info("Using custom thresholds from config.")
            self.threshold_optimizer.set(self.custom_thresholds, n_classes=y_true_thr.shape[1])
        else:
            logger.info(f"Computing optimal thresholds using dataset '{self.thresholds_ds_name}'...")
            self.threshold_optimizer.fit(y_true_thr, y_probs_thr)

        opt_thresholds = self.threshold_optimizer.thresholds
        self.visualizer.plot_thresholds(opt_thresholds, min_thr=self.min_thr, max_thr=self.max_thr)

        for ds_name, dataset in evaluation_dict.items():
            if dataset:
                metrics = self._evaluate(dataset, ds_name)
                total_metrics.update(metrics)

        return total_metrics, opt_thresholds

    def _get_probabilities(self, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.array(dataset["label"])
        y_probs = []
        for text_list in dataset["text"]:
            probs = self._predict_proba(text_list)
            y_probs.append(probs)
        return y_true, np.array(y_probs)

    def _predict_proba(self, text_list: list[str]) -> np.ndarray:
        probs = self.model.predict_proba(text_list).numpy()[:, :, 1]
        return np.max(probs, axis=0)

    def _evaluate(self, dataset: Dataset, dataset_name: str) -> dict:
        y_true, y_probs = self._get_probabilities(dataset)
        preds = {"default": (y_probs >= 0.5).astype(int), "opt": self.threshold_optimizer.apply(y_probs)}
        self.visualizer.plot_all(
            y_true=y_true,
            y_probs=y_probs,
            y_pred_before=preds["default"],
            y_pred_after=preds["opt"],
            title_prefix=dataset_name,
        )

        langs = np.array(dataset["lang"])
        feature_names = self.valid_feature_names if self.plot_feature else None

        metrics = {}
        for name, y_pred in preds.items():
            if name == "opt":
                ds_name = f"{dataset_name}_{name}"
            else:
                ds_name = dataset_name
            eval_data = EvalData(
                y_true=y_true,
                y_pred=y_pred,
                langs=langs,
                feature_names=feature_names,
                dataset_name=ds_name,
            )
            metrics.update(compute_all_metrics(eval_data))

        logger.info(
            f"Evaluated dataset '{dataset_name}': default vs optimized metrics computed ({self.thresholds_metric})."
        )
        return metrics
