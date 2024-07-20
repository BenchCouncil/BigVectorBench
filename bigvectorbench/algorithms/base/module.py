""" Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking. """
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil

import numpy as np

class BaseANN(object):
    """
    Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking.
    """

    @property
    def num_entities(self) -> int:
        """
        Get number of entities
        """
        return self._num_entities

    @num_entities.setter
    def num_entities(self, value: int) -> None:
        self._num_entities = value

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""

    def get_memory_usage(self) -> Optional[float]:
        """
        Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """
        return psutil.Process().memory_info().rss / 1024

    def load_data(
            self,
            embeddings: np.array,
            labels: np.ndarray | None = None,
            label_names: list[str] | None = None,
            label_types: list[str] | None = None,
            ) -> None:
        """
        Fit the ANN algorithm to the provided data

        Args:
            embeddings (np.array): embeddings
            labels (np.array): labels
            label_names (list[str]): label names
            label_types (list[str]): label types
        """

    def create_index(self) -> None:
        """
        Create index for the ANN algorithm
        """

    def query(
            self,
            v : np.ndarray,
            n : int,
            filter_expr = None
            ) -> list[int]:
        """
        Performs a query on the algorithm to find the nearest neighbors

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            filter_expr (str): The search expression

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices

    def batch_query(
            self,
            vectors: np.ndarray,
            n: int,
            exprs: list | None = None
            ) -> None:
        """
        Performs multiple queries at once and lets the algorithm figure out how to handle it.

        The default implementation uses a ThreadPool to parallelize query processing.

        Args:
            vectors (np.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
            exprs (list[str]): The search expressions for each query.

        Returns:
            None: self.get_batch_results() is responsible for retrieving batch result
        """
        pool = ThreadPool()
        if exprs is None:
            self.res = pool.map(lambda q: self.query(q, n), vectors)
        else:
            self.res = pool.starmap(lambda q, e: self.query(q, n, e), zip(vectors, exprs))

    def get_batch_results(self) -> np.array:
        """
        Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        """
        Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def insert(
        self,
        embeddings : np.ndarray,
        labels : np.ndarray | None = None
    ) -> None:
        """
        Single insert data

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """

    def update(
        self,
        index : int,
        embeddings : np.ndarray,
        labels : np.ndarray | None = None
    ) -> None:
        """
        Single update data

        Args:
            index (int): index to update
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """

    def delete(
        self,
        index: int,
    ) -> None:
        """
        Single delete data

        Args:
            index (int): index to delete

        Returns:
            None
        """

    def __str__(self) -> str:
        return self.name
