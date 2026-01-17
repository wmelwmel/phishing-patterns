from collections import Counter

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.datamodels import LoadDFResult
from src.labeling.process_labeled_data import create_binary_vector, get_matched_rows
from src.utils import get_config


class ManualDatasetHandler:
    def __init__(self, features: list[dict]) -> None:
        self.feature_names = [feature["name"].strip().lower() for feature in features]

    def get_manual_eval_df(self, labeled_df: pd.DataFrame, manual_labels_name: str) -> LoadDFResult:
        manual_labels = get_config(manual_labels_name)
        self._validate_manual_labels(manual_labels)

        logger.info("=" * 50)
        logger.info("PROCESSING DATASET")
        logger.info("=" * 50)

        matched_rows = get_matched_rows(labeled_df, manual_labels)

        matched_indices = list(matched_rows.keys())
        manual_eval_df = labeled_df.loc[matched_indices].copy()
        llm_res_vec = manual_eval_df.res_vector.to_numpy()

        labeled_df_filtered = labeled_df.drop(matched_indices)

        logger.info(f"Files matched: {len(manual_eval_df)}/{len(manual_labels)}")

        if missing_keys := set(manual_labels) - set(matched_rows.values()):
            logger.warning(f"Missing files in dataset ({len(missing_keys)}):")
            for i, key in enumerate(missing_keys, 1):
                logger.warning(f"  {i}. {key}")

        manual_eval_df["res_vector"] = manual_eval_df.index.map(
            lambda idx: create_binary_vector(manual_labels[matched_rows[idx]], self.feature_names)
        )
        manual_res_vec = manual_eval_df.res_vector.to_numpy()

        all_applied_labels = [
            label.strip().lower() for idx in manual_eval_df.index for label in manual_labels[matched_rows[idx]]
        ]

        logger.info("=" * 50)
        logger.info("MANUAL DATASET STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Files: {len(manual_eval_df)}")

        self._get_value_counts(all_applied_labels, "in final dataset")
        logger.info(f"Total labels applied: {len(all_applied_labels)}")

        return LoadDFResult(
            labeled_df_filtered=labeled_df_filtered,
            manual_eval_df=manual_eval_df,
            llm_res_vec=llm_res_vec,
            manual_res_vec=manual_res_vec,
        )

    def _validate_manual_labels(self, manual_labels: DictConfig) -> None:
        all_labels = [label.strip().lower() for labels in manual_labels.values() for label in labels]

        logger.info("=" * 50)
        logger.info("MANUAL LABELS VALIDATION")
        logger.info("=" * 50)
        logger.info(f"Total files: {len(manual_labels)}")

        feature_counter = self._get_value_counts(all_labels, "in manual labels")
        self._check_features(feature_counter)
        logger.info(f"Total labels: {sum(feature_counter.values())}")

    def _get_value_counts(self, feature_labels: list[str], context: str = "") -> Counter[str]:
        feature_counter = Counter(feature_labels)

        if not feature_counter:
            logger.info(f"No features found {context}")
            return feature_counter

        max_item = feature_counter.most_common(1)[0]
        min_item = min(feature_counter.items(), key=lambda x: x[1])

        logger.info(f"Feature distribution {context}:")
        logger.info(f"Total features: {len(feature_counter)}")
        logger.info(f"Max: '{max_item[0]}' = {max_item[1]}")
        logger.info(f"Min: '{min_item[0]}' = {min_item[1]}")
        logger.info("Count details (descending order):")

        for feature, count in feature_counter.most_common():
            logger.info(f"  {feature}: {count}")

        return feature_counter

    def _check_features(self, feature_counter: Counter[str]) -> None:
        invalid_features = set(feature_counter) - set(self.feature_names)
        if invalid_features:
            logger.warning(f"Found {len(invalid_features)} unexpected names: {invalid_features}")
        else:
            logger.info("All features are valid")
