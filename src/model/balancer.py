from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm


class DataBalancer:
    def __init__(
        self,
        res_col: str,
        target_count: None | int,
        verbose: bool = True,
        mode: Literal["kmeans", "maxmin"] = "kmeans",
        random_seed: int = 42,
    ) -> None:
        self.mode = mode
        self.res_col = res_col
        self.target_count = target_count
        self.verbose = verbose
        self.random_seed = random_seed

    def get_balanced_df(self, lang_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        balanced_lang_dfs = []
        for lang_df in lang_dfs:
            balanced_lang_dfs.append(self._balance(lang_df.reset_index(drop=True)))
        balanced_df = pd.concat(balanced_lang_dfs)
        return balanced_df

    def _balance(self, df: pd.DataFrame) -> pd.DataFrame:
        balanced_df = pd.DataFrame()
        feature_matrix = np.vstack(df[self.res_col].values)
        feature_counts = np.sum(feature_matrix, axis=0)
        total_zero_rows = (feature_matrix.sum(axis=1) == 0).sum()
        sorted_features = np.argsort(feature_counts)

        n_features = len(feature_counts)
        target_count = self.target_count if self.target_count is not None else feature_counts.min()

        if self.verbose:
            logger.info(f"Starting balance: min={feature_counts.min()}, max={feature_counts.max()}")
            logger.info(f"Target count per feature: {target_count}")
            logger.info(f"Total zero-vector rows in original df: {total_zero_rows}")

        for feat_idx in tqdm(sorted_features, desc=f"Balancing features (target: {target_count})"):
            mask = feature_matrix[:, feat_idx] == 1
            balanced_df = self._process_feature_mask(df, balanced_df, mask, target_count)

        zero_mask = feature_matrix.sum(axis=1) == 0
        if zero_mask.any():
            balanced_df = self._process_feature_mask(df, balanced_df, zero_mask, target_count)

        if self.verbose:
            self._log_balanced_stat(
                n_features, target_count, balanced_df, feature_counts, total_zero_rows=total_zero_rows
            )

        return balanced_df

    def _process_feature_mask(
        self, df: pd.DataFrame, balanced_df: pd.DataFrame, mask: np.ndarray, target_count: int
    ) -> pd.DataFrame:
        available_data = df[mask].copy()

        if not balanced_df.empty:
            existing_indices = set(balanced_df.index) & set(df[mask].index)
            available_data = available_data.loc[~available_data.index.isin(existing_indices)]
            existing_count = len(existing_indices)
        else:
            existing_count = 0

        if len(available_data) == 0:
            return balanced_df

        needed_count = max(0, target_count - existing_count)
        if needed_count > 0:
            selected_data = self._select_diverse_samples(available_data, needed_count)
            balanced_df = pd.concat([balanced_df, selected_data])

        return balanced_df

    def _select_diverse_samples(self, data: pd.DataFrame, needed_count: int) -> pd.DataFrame:
        if len(data) <= needed_count:
            return data

        embeddings = np.vstack(data.emb.values)

        if self.mode == "kmeans":
            selected_indices = self._select_diverse_kmeans(embeddings, needed_count)
        elif self.mode == "maxmin":
            selected_indices = self._select_diverse_maxmin(embeddings, needed_count)
        else:
            msg = f"Invalid mode '{self.mode}'. Must be 'kmeans' or 'maxmin'."
            logger.error(msg)
            raise ValueError(msg)

        return data.iloc[selected_indices]

    def _select_diverse_kmeans(self, embeddings: np.ndarray, needed_count: int) -> np.ndarray:
        kmeans = KMeans(n_clusters=needed_count, random_state=self.random_seed, n_init=10)
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_
        distances = cdist(centroids, embeddings, metric="cosine")
        _, col_ind = linear_sum_assignment(distances)
        return col_ind

    def _select_diverse_maxmin(self, embeddings: np.ndarray, needed_count: int) -> np.ndarray:
        if len(embeddings) <= needed_count:
            return np.arange(len(embeddings))

        rng = np.random.default_rng(self.random_seed)
        selected = [rng.integers(len(embeddings))]
        distances = cdist(embeddings, embeddings, metric="cosine")

        for _ in range(1, needed_count):
            min_dist_to_selected = distances[:, selected].min(axis=1)
            min_dist_to_selected[selected] = -np.inf
            next_idx = int(np.argmax(min_dist_to_selected))
            selected.append(next_idx)
        return np.array(selected)

    def _log_balanced_stat(
        self,
        feature_count: int,
        target_count: int,
        balanced_df: pd.DataFrame,
        counts: NDArray[np.int_],
        total_zero_rows: int,
    ) -> None:
        feature_matrix_bal = np.vstack(balanced_df[self.res_col].to_numpy())
        counts_bal = feature_matrix_bal.sum(axis=0)

        logger.info("Feature balancing results:")
        for i in range(feature_count):
            logger.info(f"Feature {i + 1}: {counts_bal[i]}/{target_count} (was {counts[i]})")

        zero_rows_count = (feature_matrix_bal.sum(axis=1) == 0).sum()
        logger.info(f"Zero-vector: {zero_rows_count}/{target_count} (was {total_zero_rows})")

        below = sum(counts_bal < target_count)
        above = sum(counts_bal > target_count)
        logger.info(f"Summary: {below} features below target, {above} features above target")
        logger.info(f"Final dataset size: {len(balanced_df)} rows")
