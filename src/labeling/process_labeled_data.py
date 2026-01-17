import re

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from omegaconf import DictConfig
from tqdm import tqdm

from src.paths import labeled_df_path, manual_init_df_path, sentences_dir
from src.prepare_data.data_processing import (
    clean_sentence_col,
    get_sentence_df,
)
from src.utils import get_config


def filter_not_valid_clusters(
    df_filter: pd.DataFrame, analysis_results: dict, feature_vec_len: int = 21
) -> pd.DataFrame:
    not_valid_len_clusters = []
    not_valid_value_clusters = []
    df_filter["cluster"] = df_filter["cluster"].astype(str)
    for cluster_id, cluster_results in tqdm(analysis_results.items()):
        features_vector = np.array(cluster_results["features"])

        if ~np.all((features_vector == 0) | (features_vector == 1)):
            not_valid_value_clusters.append(cluster_id)
        elif len(features_vector) != feature_vec_len:
            not_valid_len_clusters.append(cluster_id)

    not_valid_clusters = not_valid_len_clusters + not_valid_value_clusters
    not_valid_clusters_n = len(not_valid_clusters)
    logger.info(
        f"Found {not_valid_clusters_n} ({not_valid_clusters_n / len(analysis_results) * 100:.2f}%) invalid clusters"
    )

    sentences_init_count = len(df_filter)
    df_filter = df_filter[~df_filter.cluster.isin(not_valid_clusters)]
    sentences_valid_count = len(df_filter)
    filtered_count = sentences_init_count - sentences_valid_count
    logger.info(f"Filter {filtered_count} ({filtered_count / sentences_init_count * 100:.2f}%) invalid sentences")

    return df_filter, not_valid_clusters


def get_message_id_results(df_messages: pd.DataFrame, analysis_results: dict) -> tuple[dict, dict]:
    messages_id_vectors: dict[str, np.ndarray] = {}
    messages_id_add_descriptions: dict[str, list[str]] = {}

    df_messages["message_id"] = df_messages["message_id"].astype(str)

    for cluster_id, cluster_results in tqdm(analysis_results.items()):
        msg_ids = sorted(df_messages[df_messages.cluster == str(cluster_id)].message_id.unique())
        features_vector = cluster_results["features"]
        additional_desc = cluster_results["description"]
        np_vector = np.array(features_vector, dtype=np.uint8)

        for msg_id in msg_ids:
            if msg_id in messages_id_vectors:
                np.bitwise_or(messages_id_vectors[msg_id], np_vector, out=messages_id_vectors[msg_id])
            else:
                messages_id_vectors[msg_id] = np_vector.copy()

            if msg_id in messages_id_add_descriptions:
                messages_id_add_descriptions[msg_id].extend([additional_desc])
            else:
                messages_id_add_descriptions[msg_id] = [additional_desc]

    return messages_id_vectors, messages_id_add_descriptions


def get_labeled_results(
    df_init: pd.DataFrame,
    messages_id_vectors: dict[str, np.ndarray],
    messages_id_add_descriptions: dict[str, list[str]],
    save_df: bool = True,
) -> pd.DataFrame:
    df_init["res_vector"] = df_init.index.astype(str).map(messages_id_vectors)
    df_init["add_desc"] = df_init.index.astype(str).map(messages_id_add_descriptions)
    df_init["vector_sum"] = df_init["res_vector"].apply(
        lambda x: sum(x) if (x is not None and not isinstance(x, (float, int))) else 0
    )
    df_results = df_init[["file_path", "label", "sentence", "res_vector", "add_desc", "vector_sum"]].sort_values(
        by="vector_sum", ascending=False
    )
    df_results["full_text"] = df_results["sentence"].apply(lambda x: "\n ".join(x))

    if save_df:
        df_results.to_pickle(labeled_df_path)
        logger.info(f"Labeled results saved to: {labeled_df_path}")

    return df_results


def get_matched_rows(labeled_df: pd.DataFrame, manual_labels: DictConfig) -> dict[int, str]:
    file_paths_lower = labeled_df["file_path"].str.lower()
    manual_keys_lower = [str(k).lower() for k in manual_labels]
    pattern = "|".join(map(re.escape, manual_keys_lower))
    matched_series = file_paths_lower.str.extract(f"({pattern})", expand=False)
    return matched_series.dropna().to_dict()


def create_binary_vector(labels: list[str], feature_names: list[str]) -> NDArray[np.int_]:
    normalized_labels = [label.strip().lower() for label in labels]
    normalized_feature_names = [f_name.strip().lower() for f_name in feature_names]
    return np.array([1 if feat in normalized_labels else 0 for feat in normalized_feature_names])


def get_labeled_names(bin_vector: np.ndarray, feature_names: list[str]) -> list[str]:
    return [feature_names[i] for i, idx in enumerate(bin_vector) if idx >= 1]


def get_golden_set(
    update: bool = True,
    label_config_name: str = "email_patterns_manual_labels.yaml",
    include_columns: list[str] = ["file_path", "label", "sentence", "lang"],
) -> pd.DataFrame:
    if update:
        logger.info("Building golden set from raw sentences...")
        df_sentences = get_sentence_df(sentences_dir)
        manual_labels = get_config(label_config_name)
        features = get_config("features.yaml")["features"]

        matched_rows = get_matched_rows(df_sentences, manual_labels)
        matched_indices = list(matched_rows.keys())

        golden_df = df_sentences.loc[matched_indices].copy()[include_columns]
        golden_df["sentence"] = clean_sentence_col(golden_df["sentence"])

        feature_names = [feature["name"] for feature in features]
        golden_df["res_vector"] = golden_df.index.map(
            lambda idx: create_binary_vector(manual_labels[matched_rows[idx]], feature_names)
        )
        golden_df.to_pickle(manual_init_df_path)
        logger.info(f"Golden set saved to {manual_init_df_path}: {len(golden_df)} rows")
    else:
        logger.info(f"Loading golden set from {manual_init_df_path}")

        if not manual_init_df_path.exists():
            msg = f"Golden set file not found: {manual_init_df_path}. Run with update=True to rebuild."
            logger.error(msg)
            raise FileNotFoundError(msg)

        golden_df = pd.read_pickle(manual_init_df_path)
        logger.info(f"Golden set loaded: {len(golden_df)} rows")
    return golden_df
