from enum import Enum

import numpy as np
import pandas as pd
from datasets import Dataset
from numpy.typing import NDArray
from plotly.graph_objects import Figure
from pydantic import BaseModel, ConfigDict


class ClusteringResult(BaseModel):
    clusters: NDArray[np.int_]
    unique_clusters: NDArray[np.int_]
    n_clusters: int
    noise_count: int
    index_score: float | None
    eps: float
    min_samples: int
    noise_percentage: float
    weighted_score: float

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class DatasetsResult(BaseModel):
    train_dataset: Dataset
    test_dataset: Dataset
    full_test_dataset: Dataset | None
    manual_eval_dataset: Dataset | None
    golden_dataset: Dataset | None
    valid_features: list[int]
    llm_res_vec: np.ndarray | None
    manual_res_vec: np.ndarray | None
    matrix_fig: Figure

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class LoadDFResult(BaseModel):
    labeled_df_filtered: pd.DataFrame | list[pd.DataFrame]
    manual_eval_df: pd.DataFrame | None
    llm_res_vec: np.ndarray | None
    manual_res_vec: np.ndarray | None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class EvalData(BaseModel):
    y_true: NDArray[np.int_]
    y_pred: NDArray[np.int_]
    langs: NDArray[np.str_] | None
    feature_names: list[str] | None
    dataset_name: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ThresholdsMetric(str, Enum):
    PRECISION = "precision"
    RECALL = "recall"
    ACCURACY = "accuracy"
    FBETA = "fbeta"
    F1 = "f1"
