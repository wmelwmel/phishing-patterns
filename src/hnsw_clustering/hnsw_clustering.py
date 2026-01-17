import numpy as np
from numpy.typing import NDArray

from src.datamodels import ClusteringResult


class HNSWClustering:
    def __init__(self, all_neighbors: NDArray[np.int64], all_distances: NDArray[np.float32]):
        """
        Initializes the HNSWClustering class with neighbors and distances data.

        :param all_neighbors: Indices of the nearest neighbors for each embedding.
        :param all_distances: Distances to the nearest neighbors for each embedding.
        """
        self.all_neighbors = all_neighbors
        self.all_distances = all_distances

    def cluster(self, eps: float, min_samples: int) -> ClusteringResult:
        """
        Performs density-based clustering using HNSW nearest neighbors results.

        :param eps: DBSCAN parameter eps.
        :param min_samples: DBSCAN parameter min_samples.
        :return: ClusteringResult
        """
        num_elements = self.all_neighbors.shape[0]

        labels = np.full(num_elements, -1, dtype=int)
        visited = np.zeros(num_elements, dtype=bool)
        cluster_id = 0

        for idx in range(num_elements):
            if visited[idx]:
                continue

            stack = [idx]
            cluster = {idx}
            visited[idx] = True
            labels[idx] = cluster_id

            while stack:
                point = stack.pop()
                neighbors = self.all_neighbors[point]
                distances = self.all_distances[point]
                close_neighbors = neighbors[distances <= eps]

                for neighbor in close_neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        labels[neighbor] = cluster_id
                        cluster.add(neighbor)
                        stack.append(neighbor)

            if len(cluster) < min_samples:
                for point in cluster:
                    labels[point] = -1
            else:
                cluster_id += 1

        noise_count = (labels == -1).sum()
        noise_pct = noise_count / len(labels)

        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

        return ClusteringResult(
            clusters=labels,
            unique_clusters=unique_clusters,
            n_clusters=n_clusters,
            noise_count=noise_count,
            index_score=None,
            eps=eps,
            min_samples=min_samples,
            noise_percentage=noise_pct,
            weighted_score=0.0,
        )
