import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.datamodels import ClusteringResult
from src.hnsw_clustering.hnsw_clustering import HNSWClustering
from src.hnsw_clustering.hnsw_index import HNSWIndex

warnings.filterwarnings("ignore")


class ClusteringHandler:
    def __init__(self, config: DictConfig, seed: int = 42) -> None:
        self.hnsw_params = config.hnsw_params
        self.clustering_params = config.clustering_params
        self.seed = seed

    def run_clustering(
        self,
        embeddings: NDArray[np.float32],
        index_dir: Path,
    ) -> tuple[ClusteringResult, pd.DataFrame]:
        """
        Performs clustering on embeddings using an HNSW index.

        :param embeddings: Embeddings used to build the index.
        :param index_dir: HNSW index directory.
        :return: ClusteringResult
        """
        eps = self.clustering_params.eps
        min_samples = self.clustering_params.min_samples
        auto_optimize = self.clustering_params.auto_optimize
        eps_range = self.clustering_params.eps_range
        eps_step = self.clustering_params.eps_step
        min_samples_range = self.clustering_params.min_samples_range
        top_n_results = self.clustering_params.top_n_results
        recalculate_index = self.clustering_params.recalculate_index

        logger.info("Init hnsw index...")
        hnsw_index = self._init_hnsw_index(embeddings, index_dir, recalculate_index)

        hnsw_clustering = self._get_hnsw_clustering(hnsw_index, embeddings)

        results_df = None

        if auto_optimize or eps is None or min_samples is None:
            logger.info("Searching for optimal clustering parameters...")
            top_results, results_df = self._find_optimal_params(
                hnsw_clustering, embeddings, eps_range, eps_step, min_samples_range, top_n=top_n_results
            )

            if not top_results:
                raise ValueError("No valid clustering parameters found")

            best_result = top_results[0]
            logger.info("Top parameters found:")
            logger.info(f"\n{results_df.to_string(index=False)}")
            logger.info(f"Selected parameters: eps={best_result.eps:.2f}, min_samples={best_result.min_samples}")

        else:
            logger.info("Clustering...")

            best_result = self._perform_clustering(
                hnsw_clustering,
                embeddings,
                eps,
                min_samples,
            )

            logger.info("Clustering completed.")

            if not best_result:
                raise ValueError("Clustering failed with given parameters")

        logger.info("Clustering results with selected parameters:")
        logger.info(f"Parameters: eps={best_result.eps:.2f}, min_samples={best_result.min_samples}")
        logger.info(f"Clusters found: {best_result.n_clusters}")
        logger.info(f"Noise points: {best_result.noise_count} ({best_result.noise_percentage:.2%})")
        logger.info(f"Total points: {len(embeddings)}")

        if best_result.index_score is not None:
            logger.info(f"Index score: {best_result.index_score:.3f}")
            logger.info(f"Weighted score: {best_result.weighted_score:.3f}")
        else:
            logger.info("Index score unavailable (requires >1 valid cluster)")

        return best_result, results_df

    def _init_hnsw_index(self, embeddings: NDArray[np.float32], index_dir: Path, recalculate_index: bool) -> HNSWIndex:
        hnsw_index = HNSWIndex(self.hnsw_params)
        index_path = index_dir / self.hnsw_params.hnsw_index_name
        if index_path.exists() and not recalculate_index:
            logger.info("Loading HNSW index...")
            hnsw_index.load_index(index_path, dim=embeddings.shape[1])
            logger.info(f"HNSW index loaded from {index_path}")
        else:
            logger.info("Building HNSW index...")
            hnsw_index.build_index(embeddings)
            hnsw_index.save_index(index_path)
            logger.info(f"HNSW index saved to {index_path}")
        return hnsw_index

    def _get_hnsw_clustering(self, hnsw_index: HNSWIndex, embeddings: NDArray[np.float32]) -> HNSWClustering:
        logger.info("Calculating all neighbor distances...")
        all_neighbors, all_distances = hnsw_index.query_index(embeddings)

        logger.info("HNSWClustering initialization...")
        hnsw_clustering = HNSWClustering(all_neighbors, all_distances)
        return hnsw_clustering

    def _find_optimal_params(
        self,
        hnsw_clustering: HNSWClustering,
        embeddings: NDArray[np.float32],
        eps_range: tuple[float, float] = (0.1, 1.0),
        eps_step: float = 0.05,
        min_samples_range: tuple[int, int] = (3, 10),
        top_n: int = 10,
    ) -> tuple[list[ClusteringResult], pd.DataFrame]:
        results = []
        eps_values = np.arange(eps_range[0], eps_range[1], eps_step)

        for eps in tqdm(eps_values, desc="Scanning optimal values"):
            for min_samples in range(min_samples_range[0], min_samples_range[1]):
                result = self._perform_clustering(
                    hnsw_clustering,
                    embeddings,
                    eps,
                    min_samples,
                )
                if result:
                    results.append(result)

        results.sort(key=lambda x: x.weighted_score, reverse=True)
        top_results: list[ClusteringResult] = results[:top_n]

        df_results = pd.DataFrame(
            [
                {
                    "eps": r.eps,
                    "min_samples": r.min_samples,
                    "n_clusters": r.n_clusters,
                    "noise_count": r.noise_count,
                    "noise_percentage": r.noise_percentage,
                    "index_score": r.index_score,
                    "weighted_score": r.weighted_score,
                }
                for r in top_results
            ]
        )

        return top_results, df_results

    def _perform_clustering(
        self, hnsw_clustering: HNSWClustering, embeddings: NDArray[np.float32], eps: float, min_samples: int
    ) -> ClusteringResult:
        cr = hnsw_clustering.cluster(eps, min_samples)
        cr_upd = self._calculate_weighted_score(cr, embeddings)
        return cr_upd

    def _calculate_weighted_score(
        self, cr: ClusteringResult, embeddings: NDArray[np.float32], metric: str = "cosine"
    ) -> ClusteringResult:
        max_samples_per_cluster = self.clustering_params.max_samples_per_cluster
        if cr.n_clusters < 2 or cr.noise_count == len(embeddings):
            return cr.model_copy()

        labels = cr.clusters
        unique_clusters = [c for c in np.unique(labels) if c != -1]

        sampled_idx = []
        rng = np.random.default_rng(seed=self.seed)

        for cluster_id in unique_clusters:
            cluster_idx = np.where(labels == cluster_id)[0]
            if len(cluster_idx) > max_samples_per_cluster:
                sampled = rng.choice(cluster_idx, size=max_samples_per_cluster, replace=False)
            else:
                sampled = cluster_idx
            sampled_idx.extend(sampled.tolist())

        if len(sampled_idx) < 2:
            return cr.model_copy()

        sampled_np_idx = np.array(sampled_idx)
        sampled_embeddings = embeddings[sampled_np_idx]
        sampled_labels = labels[sampled_np_idx]

        try:
            index_score = silhouette_score(sampled_embeddings, sampled_labels, metric=metric)
        except Exception as e:
            index_score = None
            logger.exception(f"Error computing index_score: {e}")

        weighted_score = index_score * (1 - cr.noise_percentage) if index_score is not None else 0.0

        return cr.model_copy(update={"index_score": index_score, "weighted_score": weighted_score})
