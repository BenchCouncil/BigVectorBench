import numpy as np
import sklearn.neighbors

from bigvectorbench.distance import metrics as pd
from bigvectorbench.algorithms.base.module import BaseANN


class BruteForce(BaseANN):
    def __init__(self, metric):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[
            self._metric
        ]
        self._nbrs = sklearn.neighbors.NearestNeighbors(
            algorithm="brute", metric=metric
        )
        self.name = "BruteForce()"
        self.embeddings = None
        self.labels = None
        self.label_names = None
        self.label_types = None

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        self._nbrs.fit(embeddings)
        self.embeddings = embeddings
        self.labels = labels
        self.label_names = label_names
        self.label_types = label_types
        self.num_entities = len(embeddings)

    def query(self, v, n, filter_expr=None):
        # if filter_expr is not None:
        #     raise NotImplementedError("BruteForce doesn't support filtering")
        if filter_expr is not None:
            ann_res = list(
                self._nbrs.kneighbors(
                    [v], return_distance=False, n_neighbors=len(self.labels)
                )[0]
            )
            satisfied_indices = []
            for i in ann_res:
                label = self.labels[i]
                label_str = (
                    ",".join(self.label_names) + " = " + ",".join(str(x) for x in label)
                )
                exec(label_str)
                if eval(filter_expr) == True:
                    satisfied_indices.append(i)
                if len(satisfied_indices) == n:
                    break
            return satisfied_indices
        else:
            return list(
                self._nbrs.kneighbors([v], return_distance=False, n_neighbors=n)[0]
            )

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors(
            [v], return_distance=True, n_neighbors=n
        )
        return zip(list(positions[0]), list(distances[0]))

    def insert(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.embeddings = np.append(self.embeddings, embeddings[np.newaxis, :], axis=0)
        if labels is not None:
            self.labels = np.append(self.labels, labels[np.newaxis, :], axis=0)
        self._nbrs.fit(self.embeddings)
        self.num_entities += 1

    def update(
        self, index: int, embeddings: np.ndarray, labels: np.ndarray | None = None
    ) -> None:
        self.embeddings[index] = embeddings
        if labels is not None:
            self.labels[index] = labels
        self._nbrs.fit(self.embeddings)

    def delete(self, index: int) -> None:
        self.embeddings[index] = [0] * len(self.embeddings[index])
        if self.labels is not None:
            self.labels[index] = [0] * len(self.labels[index])
        self._nbrs.fit(self.embeddings)
        self.num_entities -= 1

class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=np.float32):
        if metric not in ("angular", "euclidean", "hamming", "jaccard"):
            raise NotImplementedError(
                "BruteForceBLAS doesn't support metric %s" % metric
            )
        elif metric == "hamming" and precision != np.bool_:
            raise NotImplementedError(
                f"BruteForceBLAS doesn't support precision {precision} with Hamming distances"
            )
        self._metric = metric
        self._precision = precision
        self.name = "BruteForceBLAS()"
        self.label_names = None
        self.label_types = None
        self.labels = None
        self.index = None
        self.lengths = None

    def fit(
        self,
        X,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ):
        """Initialize the search index."""
        self.label_names = label_names
        self.label_types = label_types
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        if self._metric == "angular":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            # normalize index vectors to unit length
            X /= np.sqrt(lens)[..., np.newaxis]
            self.index = np.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == "hamming":
            # Regarding bitvectors as vectors in l_2 is faster for blas
            X = X.astype(np.float32)
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = np.ascontiguousarray(X, dtype=np.float32)
            self.lengths = np.ascontiguousarray(lens, dtype=np.float32)
        elif self._metric == "euclidean":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = np.ascontiguousarray(X, dtype=self._precision)
            self.lengths = np.ascontiguousarray(lens, dtype=self._precision)
        elif self._metric == "jaccard":
            self.index = X
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        self.fit(embeddings, labels, label_names, label_types)
        self.num_entities = len(embeddings)

    def query(self, v, n, filter_expr=None):
        return [index for index, _ in self.query_with_distances(v, n, filter_expr)]

    def query_with_distances(self, v, n, filter_expr=None):
        """Find indices of `n` most similar vectors from the index to query
        vector `v`."""

        if self._metric != "jaccard":
            # use same precision for query as for index
            v = np.ascontiguousarray(v, dtype=self.index.dtype)

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        if self._metric == "angular":
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)  # noqa
            dists = -np.dot(self.index, v)
        elif self._metric == "euclidean":
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = self.lengths - 2 * np.dot(self.index, v)
        elif self._metric == "hamming":
            # Just compute hamming distance using euclidean distance
            dists = self.lengths - 2 * np.dot(self.index, v)
        elif self._metric == "jaccard":
            dists = [pd[self._metric].distance(v, e) for e in self.index]
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        if filter_expr is not None:
            # satisfied_indices = []
            for i, label in enumerate(self.labels):
                label_str = (
                    ",".join(self.label_names) + " = " + ",".join(str(x) for x in label)
                )
                exec(label_str)
                if eval(filter_expr) == False:
                    # satisfied_indices.append(i)
                    dists[i] = np.inf
            # dists = [dists[i] for i in range(len(dists)) if i in satisfied_indices else np.inf]
        # partition-sort by distance, get `n` closest
        nearest_indices = np.argpartition(dists, n)[:n]
        indices = [
            idx
            for idx in nearest_indices
            if pd[self._metric].distance_valid(dists[idx])
        ]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric].distance(ep, ev))

        return map(fix, indices)

    def insert(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.index = np.append(self.index, embeddings[np.newaxis, :], axis=0)
        if labels is not None:
            self.labels = np.append(self.labels, labels[np.newaxis, :], axis=0)
        self.num_entities += 1

    def update(
        self, index: int, embeddings: np.ndarray, labels: np.ndarray | None = None
    ) -> None:
        self.index[index] = embeddings
        if labels is not None:
            self.labels[index] = labels

    def delete(self, index: int) -> None:
        self.index[index] = [0] * len(self.index[index])
        if self.labels is not None:
            self.labels[index] = [0] * len(self.labels[index])
        self.num_entities -= 1
