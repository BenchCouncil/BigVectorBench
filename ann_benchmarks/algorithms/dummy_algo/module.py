""" Dummy algorithm that returns random indices. """
import numpy as np

from ann_benchmarks.algorithms.base.module import BaseANN


class DummyAlgo(BaseANN):
    """ Dummy algorithm that returns random indices. """
    def __init__(self, metric):
        self.name = "DummyAlgo"
        self.len = 0

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        self.len = len(embeddings)

    def query(self, v, n, filter_expr=None):
        return np.random.randint(self.len, size=n)
