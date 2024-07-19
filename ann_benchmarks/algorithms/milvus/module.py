""" Milvus CPU module with FLAT, IVFFLAT, IVFSQ8, IVFPQ, HNSW, SCANN index """
import subprocess
from time import time
import numpy as np
from pymilvus import (
    DataType,
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker
)

from ann_benchmarks.algorithms.base.module import BaseANN


def metric_mapping(_metric: str):
    """
    Mapping metric type to milvus metric type

    Args:
        _metric (str): metric type
    
    Returns:
        str: milvus metric type
    """
    _metric = _metric.lower()
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    """
    Milvus Base module
    """
    def __init__(
            self,
            metric : str,
            dim : int
            ):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.start_milvus()
        self.connects = connections
        self.connects.connect("default", host='0.0.0.0', port='19530', timeout=30)
        print(f"[Milvus] Milvus version: {utility.get_server_version()}")
        self.collection_name = "test_milvus"
        self.collection = None
        self.load_batch_size = 1000
        self.query_batch_size = 1000
        self.num_labels = 0
        self.search_params = {
            "metric_type": self._metric_type
        }
        self.name = f"Milvus metric:{self._metric}"
        if utility.has_collection(self.collection_name):
            print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
            utility.drop_collection(self.collection_name)
        self.query_vector = None
        self.query_topk = 0
        self.query_expr = None
        self.prepare_query_results = None
        self.batch_query_vectors = None
        self.batch_query_exprs = None
        self.batch_results = []
        self.batch_latencies = []
        self.requests = []

    def start_milvus(self) -> None:
        """
        Start milvus cpu standalone docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            subprocess.run(["docker", "compose", "up", "-d"], check=True)
            print("[Milvus] docker compose up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Milvus] docker compose up failed: {e}!!!")

    def stop_milvus(self) -> None:
        """
        Stop milvus cpu standalone docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            print("[Milvus] docker compose down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Milvus] docker compose down failed: {e}!!!")

    def create_collection(
            self,
            num_labels : int = 0,
            label_names : list[str] | None = None,
            label_types : list[str] | None = None,
            num_vectors : int = 1
            ) -> None:
        """
        Create collection with schema
        Args:
            num_labels (int): number of labels
            label_names (list[str]): label names
            label_types (list[str]): label types
        """
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        if num_vectors == 1:
            filed_vec = FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dim
            )
            fields = [filed_id, filed_vec]
        else:
            fields = [filed_id]
            for i in range(num_vectors):
                fields.append(
                    FieldSchema(
                        name=f"vector{i}",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self._dim
                    )
                )
        self.num_labels = num_labels
        if num_labels > 0:
            label_type_to_dtype = {
                "BOOL": DataType.BOOL,
                "INT": DataType.INT32,
                "INT8": DataType.INT8,
                "INT16": DataType.INT16,
                "INT32": DataType.INT32,
                "INT64": DataType.INT64,
                "FLOAT": DataType.FLOAT,
                "DOUBLE": DataType.DOUBLE,
                "STRING": DataType.STRING,
                "VARCHAR": DataType.VARCHAR,
                "ARRAY": DataType.ARRAY,
                "JSON": DataType.JSON,
                "BINARY_VECTOR": DataType.BINARY_VECTOR,
                "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
                "FLOAT16_VECTOR": DataType.FLOAT16_VECTOR,
                "BFLOAT16_VECTOR": DataType.BFLOAT16_VECTOR,
                "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
                "UNKNOWN": DataType.UNKNOWN,
            }
            for i in range(num_labels):
                fields.append(
                    FieldSchema(
                        name=label_names[i],
                        dtype=label_type_to_dtype.get(label_types[i].upper(), DataType.UNKNOWN)
                    )
                )
        schema = CollectionSchema(
            fields = fields,
            description = "Test milvus search",
        )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Milvus] Create collection {self.collection.describe()} successfully!!!")

    def insert_data(
            self,
            embeddings : np.ndarray,
            labels : np.ndarray | None = None
            ) -> None:
        """
        Insert embeddings and labels into collection

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels
        """
        num_vectors = embeddings.shape[1] if len(embeddings.shape) == 3 else 1
        if labels is not None:
            num_labels = len(labels[0])
            print(f"[Milvus] Insert {len(embeddings)} data with {num_labels} labels  into collection {self.collection_name}...")
        else:
            print(f"[Milvus] Insert {len(embeddings)} data into collection {self.collection_name}...")
        for i in range(0, len(embeddings), self.load_batch_size):
            batch_data = embeddings[i : min(i + self.load_batch_size, len(embeddings))]
            entities = [
                [i for i in range(i, min(i + self.load_batch_size, len(embeddings)))],
                # batch_data.tolist()
                ]
            if num_vectors == 1:
                entities.append(
                    batch_data.tolist()
                )
            else:
                for j in range(num_vectors):
                    entities.append(
                        [v[j] for v in batch_data]
                    )
            if labels is not None:
                batch_labels = labels[i : min(i + self.load_batch_size, len(embeddings))]
                for j in range(num_labels):
                    entities.append(
                        [l[j] for l in batch_labels]
                    )
            self.collection.insert(entities)
        self.collection.flush()
        self.num_entities = self.collection.num_entities
        print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

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
        num_vectors = embeddings.shape[1] if len(embeddings.shape) == 3 else 1
        if labels is not None:
            self.create_collection(len(labels[0]), label_names, label_types)
        else:
            self.create_collection(num_vectors = num_vectors)
        self.insert_data(embeddings, labels)

    def get_index_param(self) -> dict:
        """
        Get index parameters

        Note: This is a placeholder method to be implemented by subclasses.
        """
        raise NotImplementedError()

    def load_collection(self) -> None:
        """
        Load collection
        """
        print(f"[Milvus] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")

    def create_index(self) -> None:
        """
        Create index for collection
        """
        print(f"[Milvus] Create index for collection {self.collection_name}...")
        self.collection.create_index(
            field_name = "vector",
            index_params = self.get_index_param(),
            index_name = "vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        index = self.collection.index(index_name = "vector_index")
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        print(f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")
        self.load_collection()

    def create_multi_index(
            self,
            num_vectors: int
            ) -> None:
        """
        Create multi index for collection
        """
        print(f"[Milvus] Create {num_vectors} index for collection {self.collection_name}...")
        for i in range(num_vectors):
            self.collection.create_index(
                field_name = f"vector{i}",
                index_params = self.get_index_param(),
                index_name = f"vector{i}_index"
            )
        for i in range(num_vectors):
            utility.wait_for_index_building_complete(
                collection_name = self.collection_name,
                index_name = f"vector{i}_index"
            )
            index = self.collection.index(index_name = f"vector{i}_index")
            index_progress =  utility.index_building_progress(
                collection_name = self.collection_name,
                index_name = f"vector{i}_index"
            )
            print(f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")
        print(f"[Milvus] Create {num_vectors} index for collection {self.collection_name} successfully!!!")
        self.load_collection()

    def query(
            self,
            v : np.array,
            n : int,
            filter_expr : str | None = None
            ) -> list[int]:
        """
        Performs a query on the algorithm to find the nearest neighbors

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            expr (str): The search expression

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            expr = filter_expr,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

    def prepare_query(
            self,
            v : np.array,
            n : int,
            expr : str | None = None
            ) -> None:
        """
        Prepare query

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            expr (str): The search expression
        """
        self.query_vector = v
        self.query_topk = n
        self.query_expr = expr

    def run_prepared_query(self) -> None:
        """
        Run prepared query
        """
        self.prepare_query_results = self.query(self.query_vector, self.query_topk, self.query_expr)

    def get_prepared_query_results(self) -> list[int]:
        """
        Get prepared query results

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        return self.prepare_query_results

    def prepare_batch_query(
            self,
            vectors: np.ndarray,
            n: int,
            exprs: list[str] | None = None
            ) -> None:
        """
        Prepare batch query

        Args:
            vectors (np.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
            exprs (list[str]): The search expressions for each query.
        """
        self.batch_query_vectors = vectors
        self.query_topk = n
        self.batch_query_exprs = exprs

    def run_prepared_batch_query(self) -> None:
        """
        Run prepared batch query
        """
        if self.batch_query_exprs is None or len(set(self.batch_query_exprs)) == 1:
            filter_expr = self.batch_query_exprs[0] if self.batch_query_exprs else None
            for i in range(0, len(self.batch_query_vectors), self.query_batch_size):
                batch_data = self.batch_query_vectors[i : min(i + self.query_batch_size, len(self.batch_query_vectors))]
                start_time = time()
                batch_results = self.collection.search(
                    data=batch_data,
                    anns_field="vector",
                    param=self.search_params,
                    expr=filter_expr,
                    limit=self.query_topk,
                    output_fields=["id"],
                )
                for r in batch_results:
                    self.batch_results.append([e.entity.get("id") for e in r])
                self.batch_latencies.extend([(time() - start_time) / len(batch_data)] * len(batch_data))
        else:
            # Not support different exprs for batch query
            start_time = time()
            self.batch_query(self.batch_query_vectors, self.query_topk, self.batch_query_exprs)
            self.batch_latencies.extend([(time() - start_time) / len(self.batch_query_vectors)] * len(self.batch_query_vectors))
            self.batch_results = super().get_batch_results()

    def get_batch_results(self) -> list[list[int]]:
        """
        Get batch results

        Returns:
            list[list[int]]: An array of indices representing the nearest neighbors.
        """
        return self.batch_results

    def get_batch_latencies(self) -> list[float]:
        """
        Get batch latencies

        Returns:
            list[float]: An array of latencies for each query.
        """
        return self.batch_latencies

    def prepare_multi_vector_query(
            self,
            vectors: np.ndarray,
            n: int
    ):
        """
        Prepare multi vector query

        Args:
            vectors (np.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
        """
        self.query_topk = n
        self.requests.clear()
        for i, v in enumerate(vectors):
            self.requests.append(
                AnnSearchRequest(
                    data=[v],
                    anns_field=f"vector{i}",
                    param=self.search_params,
                    limit=n
            ))
        # self.rerank = RRFRanker()
        self.rerank = WeightedRanker(0.25, 0.25, 0.25, 0.25)

    def run_multi_vector_query(self) -> list[int]:
        """
        Run multi vector query

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        results = self.collection.hybrid_search(
            reqs=self.requests,
            rerank=self.rerank,
            limit=self.query_topk,
            output_fields=["id"]
        )
        # print(f"[Milvus] Multi vector query results: {results}")
        ids = [r.entity.get("id") for r in results[0]]
        distances = [r.distance for r in results[0]]
        # print(f"[Milvus] Multi vector query results: {ids} {distances}")
        return ids, distances

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
        num_vectors = embeddings.shape[0] if len(embeddings.shape) == 2 else 1
        entities = [[self.num_entities]]
        if num_vectors == 1:
            entities.append([embeddings.tolist()])
        else:
            for j in range(num_vectors):
                entities.append([embeddings[j]])
        if labels is not None:
            for j in range(self.num_labels):
                entities.append([labels[j]])
        # print(f"[Milvus] entities: {entities}")
        self.collection.insert(entities)
        # self.collection.flush()
        # self.num_entities = self.collection.num_entities
        # print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")
        self.num_entities += 1

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
        num_vectors = embeddings.shape[0] if len(embeddings.shape) == 2 else 1
        entities = [[index]]
        if num_vectors == 1:
            entities.append([embeddings.tolist()])
        else:
            for j in range(num_vectors):
                entities.append([embeddings[j]])
        if labels is not None:
            for j in range(self.num_labels):
                entities.append([labels[j]])
        # print(f"[Milvus] entities: {entities}")
        self.collection.upsert(entities)
        # self.collection.flush()

    def delete(
        self,
        index : int,
    ) -> None:
        """
        Single delete data

        Args:
            index (int): index to delete

        Returns:
            None
        """
        self.collection.delete(f"id in [{index}]")
        # self.collection.flush()

    def done(self) -> None:
        """
        Release resources
        """
        self.collection.release()
        utility.drop_collection(self.collection_name)
        self.stop_milvus()


class MilvusFLAT(Milvus):
    """ Milvus with FLAT index"""
    def __init__(
            self,
            metric : str,
            dim : int
            ):
        super().__init__(metric, dim)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(
            self,
            v : np.ndarray,
            n : int,
            filter_expr : str | None = None
            ) -> list[int]:
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            expr = filter_expr,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class MilvusIVFFLAT(Milvus):
    """ Milvus with IVF_FLAT index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
        ) -> None:
        """
        Set query arguments for IVF_FLAT index

        Args:
            nprobe (int): the number of units to query
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFSQ8(Milvus):
    """ Milvus with IVF_SQ8 index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
        ) -> None:
        """
        Set query arguments for IVF_SQ8 index

        Args:
            nprobe (int): the number of units to query
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFPQ(Milvus):
    """ Milvus with IVF_PQ index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
            ) -> None:
        """
        Set query arguments for IVF_PQ index

        Args:
            nprobe (int): the number of units to query
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFPQ metric:{self._metric}, \
            index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusHNSW(Milvus):
    """ Milvus with HNSW index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            ef : int
            ) -> None:
        """
        Set query arguments for HNSW index

        Args:
            ef (int): ef
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"MilvusHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class MilvusSCANN(Milvus):
    """ Milvus with SCANN index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
        ) -> None:
        """
        Set query arguments for IVF_SQ8 index

        Args:
            nprobe (int): the number of units to query
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"
