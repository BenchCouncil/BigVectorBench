""" Vearch GPU module for BigVectorBench framework. """

from vearch.schema.index import GPUIvfPQIndex

from bigvectorbench.algorithms.vearch.module import VearchBase

class VearchGPUIvfPQ(VearchBase):
    """Vearch GPUIvfPQ implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._ncentroids = index_param.get("ncentroids", 2048)
        self._nsubvector = index_param.get("nsubvector", 64)
        self.name = f"Vearch GPUIvfPQ metric:{self._metric}"

    def get_vector_index(self):
        """Get GPUIvfPQ vector index"""
        return GPUIvfPQIndex(
            index_name="vector_idx",
            metric_type=self._metric_type,
            ncentroids=self._ncentroids,
            nsubvector=self._nsubvector,
        )

    def set_query_arguments(self, nprobe: int = 80):
        """
        Set query arguments for weaviate query with hnsw index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "nprobe": nprobe,
            "parallel_on_queries": 0,
        }
