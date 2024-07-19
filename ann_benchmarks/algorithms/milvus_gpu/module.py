""" Milvus GPU module with CAGRA, IVFPQ, IVFFLAT, BF index """

from ann_benchmarks.algorithms.milvus.module import Milvus


def metric_mapping(_metric: str):
    """
    Mapping metric type to milvus metric type

    Args:
        _metric (str): metric type

    Returns:
        str: milvus metric type
    """
    _metric_type = {"euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class MilvusGPU_BF(Milvus):
    """ Milvus GPU Brute Force index """
    def __init__(self, metric, dim):
        super().__init__(metric, dim)
        self.name = f"MilvusGPU_BRUTE_FORCE metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "GPU_BRUTE_FORCE",
            "metric_type": self._metric_type
        }

    def query(self, v, n, expr=None):
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            expr = expr,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class MilvusGPU_IVFFLAT(Milvus):
    """ Milvus GPU IVF FLAT index """
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "GPU_IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        """
        Set query arguments for GPU_IVF_FLAT index

        Args:
            nprobe (int): the number of units to query
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusGPU_IVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusGPU_IVFPQ(Milvus):
    """ Milvus GPU IVF PQ index """
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "GPU_IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        """
        Set query arguments for GPU_IVF_PQ index

        Args:
            nprobe (int): the number of units to query
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusGPU_IVFPQ metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusGPU_CAGRA(Milvus):
    """ Milvus GPU CAGRA index """
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim)
        self._index_intermediate_graph_degree = index_param.get("intermediate_graph_degree", None)
        self._index_graph_degree = index_param.get("graph_degree", None)
        self._build_algo = index_param.get("build_algo", "IVF_PQ")

    def get_index_param(self):
        return {
            "index_type": "GPU_CAGRA",
            "params": {
                "intermediate_graph_degree": self._index_intermediate_graph_degree,
                "graph_degree": self._index_graph_degree,
                "build_algo": self._build_algo
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, itopk_size, search_width, min_iterations, max_iterations, team_size):
        """
        Set query arguments for CAGRA index

        Args:
            itopk_size (int): the size of intermediate results kept during the search.
            search_width (int): the number of entry points into the CAGRA graph during the search.
            min_iterations (int): the minimum number of iterations to run the search.
            max_iterations (int): the maximum number of iterations to run the search.
            team_size (int): the number of CUDA threads used for calculating metric distance on the GPU.
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {
                "itopk_size": itopk_size,
                "search_width": search_width,
                "min_iterations": min_iterations,
                "max_iterations": max_iterations,
                "team_size": team_size
            }
        }
        self.name = f"MilvusGPU_CAGRA metric:{self._metric}, itopk_size:{itopk_size}, search_width:{search_width}, min_iterations:{min_iterations}, max_iterations:{max_iterations}, team_size:{team_size}"
