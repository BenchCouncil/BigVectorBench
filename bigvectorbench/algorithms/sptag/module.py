""" SPTAG module for BigVectorBench framework. """

import numpy as np
import SPTAG
from bigvectorbench.algorithms.base.module import BaseANN

def metric_mapping(_metric: str):
    """
    Mapping metric type to SPTAG distance metric

    Args:
        _metric (str): metric type

    Returns:
        str: SPTAG distance metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "angular": "Cosine",
        "euclidean": "L2",
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[SPTAG] Not support metric type: {_metric}!!!")
    return _metric_type

class SPTAGBase(BaseANN):
    """SPTAG implementation"""

    def __init__(self, metric: str, dim: int):
        super().__init__()
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(metric)
        self._database_name = "SPTAG_test"

        self.num_labels = 0
        self.label_names = []
        self.load_batch_size = 1000
        self.query_batch_size = 100

        self.name = f"SPTAG Base metric:{self._metric}"
        self.search_params = None
        
        self.query_vector = None
        self.query_topk = 0
        self.query_filter = None
        self.prepare_query_results = None
        self.batch_search_queries = []
        self.batch_results = []
        self.batch_latencies = []

    def get_index_param(self) -> dict:
        """
        Get index parameters

        Note: This is a placeholder method to be implemented by subclasses.
        """
        raise NotImplementedError()

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        num_labels = len(label_names) if label_names is not None else 0
        self.num_labels = num_labels
        self.label_names = label_names
        print(f"[SPTAG] load data with {num_labels} labels!!!")
        self.num_entities = len(embeddings)
        self.index_name = "bvb_index"
        self.index_type = self.get_index_param()
        print(self.index_type)
        print(self._metric_type)
        self.index = SPTAG.AnnIndex(self.index_type, 'Float', embeddings.shape[1])
        a = self.index.SetBuildParam("NumberOfThreads", '4', "Index")
        b = self.index.SetBuildParam("DistCalcMethod", self._metric_type, "Index")
        # self.index.SetBuildParam("NumberOfThreads", '4')
        # self.index.SetBuildParam("DistCalcMethod", self._metric_type)
        # print(self.index,a,b)
        print("[SPTAG] create index successfully!!!")
        # print(embeddings.shape)
        print()
        print(f"[SPTAG] Start uploading {len(embeddings)} vectors...")
        # while not self.index.Build(embeddings.tobytes(), embeddings.shape[0], False):
        #     continue
        #     # print(f"[SPTAG] Waiting for Uploading vectors successfully!!!")
        print(embeddings.tobytes())
        ret = self.index.Build(embeddings.tobytes(), embeddings.shape[0], False)
        print(ret)
        self.index.Save(self.index_name)
        print(f"[SPTAG] Uploaded {len(embeddings)} vectors successfully!!!")
        # sleep(30)


    def create_index(self):
        """SPTAG has already started indexing when inserting data"""
        pass

    def set_query_arguments(self):
        """
        Set query arguments for SPTAG query with hnsw index
        """
        raise NotImplementedError()

    def query(self, v, n, filter_expr=None):
        if filter_expr is not None:
            raise ValueError("[SPTAG] have not supported filter-query!!!")
        j = SPTAG.AnnIndex.Load(self.index_name)
        # print(j.Search(v.tobytes(), n)[0])
        # print(j.Search(v.tobytes(), n)[1])
        # print(j.Search(v.tobytes(), n)[2])
        return j.Search(v.tobytes(), n)[0]

    def done(self):
        # shutil.rmtree(self.index_name)
        print("[SPTAG] index deleted successfully!!!")

    def insert(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        Single insert data

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """
        j = SPTAG.AnnIndex.Load(self.index_name)
        j.Add(embeddings.tobytes(), embeddings.shape[0], False)
        j.Save(self.index_name)

    # def update(
    #     self, index: int, embeddings: np.ndarray, labels: np.ndarray | None = None
    # ) -> None:
    #     """
    #     Single update data

    #     Args:
    #         index (int): index to update
    #         embeddings (np.ndarray): embeddings
    #         labels (np.ndarray): labels

    #     Returns:
    
    #         None
    #     """

    # def delete(
    #     self,
    #     index: int,
    # ) -> None:
    #     """
    #     Single delete data

    #     Args:
    #         index (int): index to delete

    #     Returns:
    #         None
    #     """
    # sptag only support delete data by vector not idx 

class SPTAGBKT(SPTAGBase):
    """SPTAG HNSW implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._nlinks = index_param.get("nlinks", 32)
        self._efConstruction = index_param.get("efConstruction", 40)

    def get_index_param(self):
        return 'BKT'


    def set_query_arguments(self, efSearch: int = 40):
        """
        Set query arguments for SPTAG query with BKT index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "efSearch": efSearch,
        }
        self.name = f"SPTAG BKT metric:{self._metric}, nlinks:{self._nlinks}, efConstruction:{self._efConstruction}, efSearch:{efSearch}"
        # self.index.SetSearchParam("MaxCheck", str(efSearch), "Index")
