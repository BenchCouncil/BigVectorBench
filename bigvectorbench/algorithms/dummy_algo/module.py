""" Dummy algorithm that returns random indices. """

import numpy as np

from bigvectorbench.algorithms.base.module import BaseANN


class DummyAlgo(BaseANN):
    """Dummy algorithm that returns random indices."""

    def __init__(self, metric):
        self.name = "DummyAlgo"
        self.len = 0
        super().__init__()

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        self.len = len(embeddings)

    def query(self, v, n, filter_expr=None):
        return np.random.choice(self.len, size=n, replace=False)
