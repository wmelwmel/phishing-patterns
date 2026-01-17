from pathlib import Path
from typing import Dict, Optional, Tuple

import hnswlib
import numpy as np
from numpy.typing import NDArray


class HNSWIndex:
    def __init__(self, hnsw_params: Dict[str, str | int]) -> None:
        """
        Initializes the HNSWIndex class with configurable parameters from dictionary.

        :param hnsw_params: Configuration dictionary with HNSW parameters.
        """
        self.space = hnsw_params["space"]
        self.M = hnsw_params["M"]
        self.ef_construction = hnsw_params["ef_construction"]
        self.ef = hnsw_params["ef"]
        self.k_neighbors = hnsw_params["k_neighbors"]
        self.index: Optional[hnswlib.Index] = None

    def build_index(self, embeddings: NDArray[np.float32]) -> None:
        """
        Builds the HNSW index from embeddings.

        :param embeddings: Embeddings to initialize the index.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy array.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2-dimensional.")

        num_elements = embeddings.shape[0]
        dim = embeddings.shape[1]
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(max_elements=num_elements, M=self.M, ef_construction=self.ef_construction, random_seed=42)
        self.index.add_items(embeddings, np.arange(num_elements))
        self.index.set_ef(self.ef)

    def save_index(self, filepath: Path) -> None:
        """
        Saves the HNSW index to disk.

        :param filepath: Full path where the index will be saved.
        """
        if not filepath.parent.exists():
            raise FileNotFoundError(f"Parent directory '{filepath.parent}' does not exist")

        if self.index is not None:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.index.save_index(str(filepath))
        else:
            raise ValueError("Index is not initialized. Build the index before saving.")

    def load_index(self, filepath: Path, dim: int) -> None:
        """
        Loads the HNSW index from disk.

        :param filepath: Full path to the saved index file.
        :param dim: Dimensionality of embeddings used in the index.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"No index file found at {filepath}. Please build and save the index first.")

        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.load_index(str(filepath))
        self.index.set_ef(self.ef)

    def query_index(
        self, query_embeddings: NDArray[np.float32], k_neighbors: Optional[int] = None
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """
        Queries the HNSW index for the k-nearest neighbors and their distances for each embedding.

        :param query_embeddings: Embeddings for which to find nearest neighbors.
        :param k_neighbors: Number of nearest neighbors to retrieve for each embedding.
                If not provided, will use the default from configuration.
        :return: Indices and distances of the nearest neighbors.
        """
        if self.index is None:
            raise ValueError("Index is not initialized. Load or build the index before querying.")

        k = k_neighbors if k_neighbors is not None else self.k_neighbors
        return self.index.knn_query(query_embeddings, k=k)
