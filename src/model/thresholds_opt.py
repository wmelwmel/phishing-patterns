from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

from datamodels import ThresholdsMetric


class ThresholdOptimizer:
    def __init__(
        self,
        metric: str = ThresholdsMetric.F1.value,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        step: float = 0.01,
    ) -> None:
        try:
            self.metric = ThresholdsMetric(metric.lower())
        except ValueError as e:
            raise ValueError(f"Unknown metric '{metric}'. Allowed values: {[m.value for m in ThresholdsMetric]}") from e

        self.min_thr = min_threshold
        self.max_thr = max_threshold
        self.step = step
        self.thresholds: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_probs: np.ndarray) -> None:
        n_classes = y_true.shape[1]
        thresholds = np.zeros(n_classes)
        metric_fn = self._get_metric_func()

        for i in range(n_classes):
            best_score, best_thr = 0.0, 0.5
            for thr in np.arange(self.min_thr, self.max_thr + self.step, self.step):
                preds = (y_probs[:, i] >= thr).astype(int)
                score = metric_fn(y_true[:, i], preds)
                if score > best_score:
                    best_score, best_thr = score, thr
            thresholds[i] = best_thr

        self.thresholds = thresholds
        logger.info(f"Optimal thresholds computed (metric={self.metric.value}): {np.round(thresholds, 3)}")

    def set(self, thresholds: list[float] | np.ndarray, n_classes: int) -> None:
        thresholds = np.array(thresholds, dtype=float)
        if len(thresholds) != n_classes:
            raise ValueError(f"Threshold length mismatch: expected {n_classes}, got {len(thresholds)}.")
        if np.any(thresholds < self.min_thr) or np.any(thresholds > self.max_thr):
            logger.warning(
                f"Some provided thresholds are outside [{self.min_thr}, {self.max_thr}] range. They will be clipped."
            )
        thresholds = np.clip(thresholds, self.min_thr, self.max_thr)
        self.thresholds = thresholds
        logger.info(f"Custom thresholds set manually: {np.round(thresholds, 3)}")

    def apply(self, y_probs: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise ValueError("Thresholds not initialized - call .fit() or .set() first.")
        return (y_probs >= self.thresholds).astype(int)

    def _get_metric_func(self) -> Any:
        logger.info(f"Apply metric {self.metric.value} to obtain new thresholds...")
        if self.metric == ThresholdsMetric.PRECISION:
            return precision_score
        elif self.metric == ThresholdsMetric.RECALL:
            return recall_score
        elif self.metric == ThresholdsMetric.ACCURACY:
            return accuracy_score
        elif self.metric == ThresholdsMetric.FBETA:
            return lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=0.5)
        return lambda y_true, y_pred: f1_score(y_true, y_pred)


class ThresholdVisualizer:
    def __init__(self, feature_names: list[str], figs_dir: str | Path = "figures/logits_thresholds") -> None:
        self.feature_names = feature_names
        self.figs_dir = Path(figs_dir)

    def plot_all(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_pred_before: np.ndarray,
        y_pred_after: np.ndarray,
        title_prefix: str = "",
    ) -> None:
        logger.info("Starting threshold visualization...")

        fig_probs = self._plot_distribution_core(y_true, y_pred_after, y_probs, "Predicted probability", title_prefix)
        mlflow.log_figure(fig_probs, str(self.figs_dir / f"{title_prefix}_probabilities_distribution.png"))

        eps = 1e-8
        y_logits = np.log((y_probs + eps) / (1 - y_probs + eps))
        fig_logits = self._plot_distribution_core(y_true, y_pred_after, y_logits, "Logit", title_prefix)
        mlflow.log_figure(fig_logits, str(self.figs_dir / f"{title_prefix}_logits_distribution.png"))

        fig_metrics = self._plot_metric_comparison(y_true, y_pred_before, y_pred_after, title_prefix)
        mlflow.log_figure(fig_metrics, str(self.figs_dir / f"{title_prefix}_metric_comparison.png"))

        logger.success("All threshold visualizations completed successfully.")

    def plot_thresholds(
        self, thresholds: np.ndarray, baseline: float = 0.5, min_thr: float = 0.0, max_thr: float = 1.0
    ) -> None:
        x = np.arange(len(self.feature_names))
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, thresholds, label="Optimized threshold")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=9
            )

        ax.axhline(baseline, color="red", linestyle="--", label=f"Baseline ({baseline})")
        ax.axhline(min_thr, color="blue", linestyle="--", label=f"Min_thr ({min_thr})")
        ax.axhline(max_thr, color="blue", linestyle="--", label=f"Max_thr ({max_thr})")
        ax.set_xlabel("Features")
        ax.set_ylabel("Threshold")
        ax.set_ylim(0, 1.1)
        ax.set_title("Optimized thresholds per feature")
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=90)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, str(self.figs_dir / "thresholds.png"))

    def _plot_distribution_core(
        self,
        y_true: np.ndarray,
        y_pred_opt: np.ndarray,
        values: np.ndarray,
        value_label: str,
        title_suffix: str,
    ) -> Figure:
        num_features = len(self.feature_names)
        n_cols = 4
        n_rows = int(np.ceil(num_features / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axs = axs.flatten()

        for i, feature in enumerate(self.feature_names):
            ax = axs[i]
            vals = values[:, i]

            mask_pos_opt = y_pred_opt[:, i] == 1
            mask_neg_opt = y_pred_opt[:, i] == 0
            mask_pos = y_true[:, i] == 1
            mask_neg = y_true[:, i] == 0

            n_pos, n_neg = mask_pos.sum(), mask_neg.sum()
            n_pos_opt, n_neg_opt = mask_pos_opt.sum(), mask_neg_opt.sum()

            if n_pos < 3:
                sns.histplot(vals[mask_neg_opt], label=f"opt_0 ({n_neg_opt})", bins=30, color="blue", alpha=0.5, ax=ax)
                sns.histplot(vals[mask_pos_opt], label=f"opt_1 ({n_pos_opt})", bins=30, color="red", alpha=0.6, ax=ax)
                sns.histplot(vals[mask_neg], label=f"init_0 ({n_neg})", bins=30, color="orange", alpha=0.5, ax=ax)
                sns.histplot(vals[mask_pos], label=f"init_1 ({n_pos})", bins=30, color="green", alpha=0.6, ax=ax)
            else:
                sns.kdeplot(vals[mask_neg_opt], label=f"opt_0 ({n_neg_opt})", fill=True, alpha=0.5, color="blue", ax=ax)
                sns.kdeplot(vals[mask_pos_opt], label=f"opt_1 ({n_pos_opt})", fill=True, alpha=0.5, color="red", ax=ax)
                sns.kdeplot(
                    vals[mask_neg], label=f"init_0 ({n_neg})", linestyle="--", fill=True, alpha=0.1, color="blue", ax=ax
                )
                sns.kdeplot(
                    vals[mask_pos], label=f"init_1 ({n_pos})", linestyle="--", fill=True, alpha=0.1, color="red", ax=ax
                )

            ax.set_title(feature)
            ax.set_xlabel(value_label)
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        fig.suptitle(f"{value_label} distributions ({title_suffix})", y=1.02, fontsize=16)
        fig.tight_layout()
        return fig

    def _plot_metric_comparison(
        self,
        y_true: np.ndarray,
        y_pred_before: np.ndarray,
        y_pred_after: np.ndarray,
        ds_name: str,
        metric_name: str = "F1-score (macro)",
    ) -> Figure:
        classes = np.arange(y_true.shape[1])
        f1_before = [f1_score(y_true[:, i], y_pred_before[:, i], average="macro") for i in classes]
        f1_after = [f1_score(y_true[:, i], y_pred_after[:, i], average="macro") for i in classes]

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.35
        ax.bar(classes - width / 2, f1_before, width, label="Before optimization")
        ax.bar(classes + width / 2, f1_after, width, label="After optimization")

        ax.set_xlabel("Features")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} per feature ({ds_name}): before vs after threshold optimization")
        ax.set_xticks(classes)
        ax.set_xticklabels(self.feature_names, rotation=90)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
