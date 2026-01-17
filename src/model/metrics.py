import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from numpy.typing import NDArray
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, fbeta_score

from src.datamodels import EvalData
from src.utils import get_config

warnings.filterwarnings("ignore")


def log_metrics(metrics: dict, dataset_name: str) -> None:
    logger.info(f"{'=' * 50}")
    logger.info(f"Metrics for {dataset_name}")
    logger.info(f"{'Metric':<20} | {'Value':<10}")
    logger.info(f"{'-' * 20}-|-{'-' * 10}")

    for metric, value in metrics.items():
        logger.info(f"{metric:<20} | {value:.4f}")
    logger.info(f"{'=' * 50}")


def get_metrics(
    y_pred: NDArray[np.int_],
    y_true: NDArray[np.int_],
    prefix: str = "",
    postfix: str = "",
    include_features: bool = False,
) -> dict:
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    tn, fp, fn, tp = None, None, None, None
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    if cm.size == 1:
        if y_true_flat[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    elif cm.size == 4:
        tn, fp, fn, tp = cm.ravel()

    metrics = {
        f"{prefix}f1_w{postfix}": f1_score(y_true, y_pred, average="weighted"),
        f"{prefix}f1_mac{postfix}": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}f0_5_mac{postfix}": fbeta_score(y_true, y_pred, beta=0.5, average="macro"),
        f"{prefix}acc{postfix}": accuracy_score(y_true, y_pred),
        f"{prefix}fp{postfix}": fp,
        f"{prefix}fn{postfix}": fn,
    }
    if include_features:
        ones = np.sum(y_true == 1)
        zeroes = np.sum(y_true == 0)
        add_metrics = {
            f"{prefix}tp/ones{postfix}": round(tp / ones, 2),
            f"{prefix}tn/zeroes{postfix}": round(tn / zeroes, 2),
            f"{prefix}ones{postfix}": ones,
            f"{prefix}zeroes{postfix}": zeroes,
        }
    else:
        add_metrics = {f"{prefix}size{postfix}": len(y_true)}
    metrics.update(add_metrics)
    return metrics


def plot_single_confusion_matrix(
    cm: NDArray,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    mlflow.log_figure(fig, filename)
    plt.close()


def plot_metrics_heatmap(
    metrics_data: dict, feature_names: list[str], metric_ds_name: str, full_artifact_path: str
) -> None:
    metrics_df = pd.DataFrame(metrics_data).T.round(3)
    n_metrics = len(metrics_df.columns)

    fig, axes = plt.subplots(1, n_metrics, figsize=(0.9 * n_metrics, max(4, len(feature_names) // 3.5)))
    fig.suptitle(f"Metrics per feature ({metric_ds_name})", fontsize=16)

    cmap = "RdYlGn"
    for i, metric in enumerate(metrics_df.columns):
        ax = axes[i] if n_metrics > 1 else axes
        values = metrics_df[metric].values.reshape(-1, 1)
        vmin = min(values)
        vmax = max(values)

        if vmin == vmax:
            if vmin == 0:
                vmin, vmax = 0, 0.1
            elif vmin == 1:
                vmin, vmax = 0.9, 1
            else:
                vmin = max(0, vmin - 0.1)
                vmax = min(1, vmax + 0.1)

        if vmin >= vmax:
            vmax = vmin + 0.1

        sns.heatmap(
            values,
            annot=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            fmt="g",
            linewidths=0.5,
            ax=ax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )

        ax.tick_params(top=True, labeltop=True)
        ax.set_xticks([0.5])
        ax.set_xticklabels([metric], rotation=0)

        if i == 0:
            ax.set_yticks(np.arange(len(feature_names)) + 0.5)
            ax.set_yticklabels(feature_names, rotation=0)
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    mlflow.log_figure(fig, f"{full_artifact_path}/feature_metrics_{metric_ds_name}.png")
    plt.close()


def plot_results(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    dataset_name: str,
    metric_ds_name: str,
    feature_names: list[str] | None = None,
    artifact_path: str = "figures",
) -> None:
    full_artifact_path = f"{artifact_path}/{dataset_name}"
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    plot_single_confusion_matrix(
        cm, f"Aggregated confusion matrix ({metric_ds_name})", f"{full_artifact_path}/agg_matrix_{metric_ds_name}.png"
    )
    if feature_names is not None:
        feature_metrics = {}
        for i, name in enumerate(feature_names):
            y_true_feature = y_true[:, i]
            y_pred_feature = y_pred[:, i]
            metrics = get_metrics(y_pred_feature, y_true_feature, include_features=True)
            feature_metrics[name] = metrics
        plot_metrics_heatmap(feature_metrics, feature_names, metric_ds_name, full_artifact_path)


def compute_metrics(
    y_pred: NDArray[np.int_],
    y_true: NDArray[np.int_],
    feature_names: list[str] | None,
    prefix: str = "",
    postfix: str = "",
) -> dict:
    metrics = get_metrics(y_pred, y_true, prefix, postfix)
    ds_name = prefix[:-1]
    metric_ds_name = ds_name + postfix
    log_metrics(metrics, metric_ds_name)
    plot_results(y_true, y_pred, ds_name, metric_ds_name, feature_names)
    return metrics


def compute_language_metrics(eval_data: EvalData, prefix: str) -> dict:
    logger.info(f"Calculating language metrics for {prefix[:-1]}...")
    langs = eval_data.langs

    unique_langs = np.unique(langs)  # type: ignore[arg-type]
    lang_metrics = {}

    for lang in unique_langs:
        lang_mask = langs == lang
        lang_y_true = eval_data.y_true[lang_mask]
        lang_y_pred = eval_data.y_pred[lang_mask]
        lang_samples = len(lang_y_true)

        if lang_samples > 0:
            postfix = f"_{lang}"
            metrics = compute_metrics(lang_y_pred, lang_y_true, eval_data.feature_names, prefix=prefix, postfix=postfix)
            lang_metrics.update(metrics)
    return lang_metrics


def compute_all_metrics(eval_data: EvalData) -> dict:
    prefix = f"{eval_data.dataset_name}_"
    dataset_metrics = compute_metrics(
        y_pred=eval_data.y_pred, y_true=eval_data.y_true, feature_names=eval_data.feature_names, prefix=prefix
    )

    if eval_data.langs is not None:
        lang_metrics = compute_language_metrics(eval_data, prefix)
        dataset_metrics.update(lang_metrics)

    return dataset_metrics


def get_labelling_metrics(
    llm_res_vec: np.ndarray,
    manual_res_vec: np.ndarray,
    langs: np.ndarray,
    valid_features: list[int],
    dataset_name: str = "llm_label",
) -> dict:
    logger.info("Labelling results evaluation...")
    valid_mask = [i for i, arr in enumerate(llm_res_vec) if isinstance(arr, np.ndarray)]
    invalid_count = len(llm_res_vec) - len(valid_mask)

    if invalid_count > 0:
        logger.warning(
            f"{invalid_count} files (out of {len(llm_res_vec)}) from manual dataset were not labeled by LLM. "
            f"Evaluation will be performed on {len(valid_mask)} files."
        )

    llm_filtered = [llm_res_vec[i] for i in valid_mask]
    manual_filtered = [manual_res_vec[i] for i in valid_mask]
    langs_filtered = np.array([langs[i] for i in valid_mask])

    if len(llm_filtered) != len(manual_filtered):
        raise ValueError(
            f"Filtered array length mismatch: "
            f"LLM has {len(llm_filtered)} files, Manual has {len(manual_filtered)} files"
        )

    if len(llm_filtered) == 0:
        raise ValueError("No valid files remaining after filtering")

    llm_matrix = np.vstack(llm_filtered)
    manual_matrix = np.vstack(manual_filtered)
    features = get_config("features.yaml")["features"]
    feature_names = [f["name"] for f in features]
    valid_feature_names = [feature_names[i] for i in valid_features]

    labeled_eval_data = EvalData(
        y_true=llm_matrix,
        y_pred=manual_matrix,
        langs=langs_filtered,
        feature_names=valid_feature_names,
        dataset_name=dataset_name,
    )

    labelling_metrics = compute_all_metrics(labeled_eval_data)
    logger.info(
        f"Successfully evaluate labelling results. Shape: {llm_matrix.shape} (LLM), {manual_matrix.shape} (Manual)"
    )

    return labelling_metrics
