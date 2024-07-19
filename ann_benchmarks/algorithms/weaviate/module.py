""" Weaviate implementation for the ANN-Benchmarks framework. """
import subprocess
import uuid
import time
import numpy as np

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, VectorDistances, Reconfigure
from weaviate.classes.query import Filter

from ann_benchmarks.algorithms.base.module import BaseANN
from ann_benchmarks.algorithms.weaviate.utils import convert_conditions_to_filters


def metric_mapping(_metric: str):
    """
    Mapping metric type to weaviate metric type

    Args:
        _metric (str): metric type

    Returns:
        str: Weaviate metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "angular": VectorDistances.COSINE,
        "euclidean": VectorDistances.L2_SQUARED,
        "hamming" : VectorDistances.HAMMING
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Weaviate] Not support metric type: {_metric}!!!")
    return _metric_type


class Weaviate(BaseANN):
    """
    Weaviate base module
    """
    def __init__(
            self,
            metric : str
        ):
        self._metric = metric
        self._metric_type = metric_mapping(metric)
        self.start_weaviate()
        time.sleep(10)
        max_tries = 10
        for _ in range(max_tries):
            try:
                self.client = weaviate.connect_to_local()
                break
            except Exception as e:
                print(f"[weaviate] connection failed: {e}")
                time.sleep(1)
        self.collection_name = "test_weaviate"
        self.collection = None
        self.num_labels = 0
        self.label_names = []
        self.name = f"Weaviate metric:{metric}"
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
        self.query_vector = None
        self.query_topk = 0
        self.query_filters = None
        self.prepare_query_results = None
        self.batch_query_vectors = None
        self.batch_query_filters = None
        self.batch_results = []
        self.batch_latencies = []

    def start_weaviate(self) -> None:
        """
        Start weaviate by docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            subprocess.run(["docker", "compose", "up", "-d"], check=True)
            print("[weaviate] docker compose up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[weaviate] docker compose up failed: {e}!!!")

    def stop_weaviate(self) -> None:
        """
        Stop weaviate by docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            print("[weaviate] docker compose down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[weaviate] docker compose down failed: {e}!!!")

    def create_collection(self, properties) -> None:
        """
        Create collection with schema
        
        Args:
            properties (list): list of properties
        """
        raise NotImplementedError

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
        print(f"[weaviate] num_labels: {num_labels}")
        # Create a collection and define properties
        properties = []
        if num_labels > 0:
            label_type_to_weaviate_type = {
                "BOOL": DataType.BOOL,
                "INT": DataType.INT,
                "INT32": DataType.INT,
                "FLOAT": DataType.NUMBER,
                "STRING": DataType.TEXT
            }
            for label_name, label_type in zip(label_names, label_types):
                properties.append(
                    Property(
                        name=label_name,
                        data_type=label_type_to_weaviate_type[label_type.upper()]
                    )
                )
        self.create_collection(properties)
        self.collection = self.client.collections.get(self.collection_name)
        print(f"[weaviate] Start loading data with {len(embeddings)} data...")
        batch_size = 1000
        print(f"[weaviate] load data with batch size: {batch_size}")
        for i in range(0, len(embeddings), batch_size):
            data_objects = []
            for j in range(i, min(i + batch_size, len(embeddings))):
                properties = {}
                if num_labels > 0:
                    for k in range(num_labels):
                        # TODO: fix if the type of labels is not int/int32
                        properties[label_names[k]] = int(labels[j][k])
                data_objects.append(
                    wvc.data.DataObject(
                        uuid=uuid.UUID(int=j),
                        properties=properties,
                        vector=embeddings[j].tolist(),
                    )
                )
            self.collection.data.insert_many(data_objects)
        print(f"[weaviate] load {self.collection.aggregate.over_all()} data successfully!!!")
        print(f"[weaviate] client.collections.list_all(simple=False): {self.client.collections.list_all(simple=False)}")
        self.num_entities = len(embeddings)

    def create_index(self) -> None:
        # Weaviate has already created the index before loading the data
        pass

    def query(self, v, n, filter_expr=None):
        ret = self.collection.query.near_vector(
            near_vector=v.tolist() if isinstance(v, np.ndarray) else v,
            limit=n,
            filters=filter_expr,
        )
        ids = [int(o.uuid) for o in ret.objects]
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
        self.query_vector = v.tolist()
        self.query_topk = n
        self.query_filters = eval(convert_conditions_to_filters(expr)) if expr is not None else None

    def run_prepared_query(self) -> None:
        """
        Run prepared query
        """
        self.prepare_query_results = self.query(self.query_vector, self.query_topk, self.query_filters)

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
        self.batch_query_filters = [eval(convert_conditions_to_filters(expr)) for expr in exprs] if exprs is not None else None

    def run_prepared_batch_query(self) -> None:
        """
        Run prepared batch query
        """
        start_time = time.time()
        if self.batch_query_filters is None:
            self.batch_query(self.batch_query_vectors, self.query_topk)
        else:
            self.batch_query(self.batch_query_vectors, self.query_topk, self.batch_query_filters)
        self.batch_latencies.extend([(time.time() - start_time) / len(self.batch_query_vectors)] * len(self.batch_query_vectors))
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
        properties={}
        if self.num_labels > 0:
            for k in range(self.num_labels):
                properties[self.label_names[k]] = int(labels[k])
        self.collection.data.insert(
            properties=properties,
            uuid=uuid.UUID(int=self.num_entities),
            vector=embeddings.tolist(),
        )
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
        properties = {}
        if self.num_labels > 0:
            for k in range(self.num_labels):
                properties[self.label_names[k]] = int(labels[k])
        self.collection.data.update(
            properties=properties,
            uuid=uuid.UUID(int=index),
            vector=embeddings.tolist(),
        )

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
        self.collection.data.delete_by_id(uuid.UUID(int=index))

    def done(self):
        self.client.close()
        self.stop_weaviate()


class WeaviateFLAT(Weaviate):
    """
    Weaviate with FLAT index
    """
    def __init__(
            self,
            metric : str,
        ):
        super().__init__(metric)
        self.name = f"WeaviateFLAT metric:{metric}"

    def create_collection(self, properties) -> None:
        self.client.collections.create(
            name=self.collection_name,
            properties=properties,
            vector_index_config=Configure.VectorIndex.flat(
                distance_metric=self._metric_type
            ),
            inverted_index_config=Configure.inverted_index()
        )


class WeaviateHNSW(Weaviate):
    """
    Weaviate with HNSW index
    """
    def __init__(
            self,
            metric : str,
            index_param: dict,
        ):
        super().__init__(metric)
        self.max_connections = index_param.get("M", None)
        self.ef_construction = index_param.get("efConstruction", None)

    def create_collection(self, properties) -> None:
        """
        Create collection with schema
        
        Args:
            properties (list): list of properties
        """
        self.client.collections.create(
            name=self.collection_name,
            properties=properties,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=self._metric_type,
                ef_construction=self.ef_construction,
                max_connections=self.max_connections
            ),
            inverted_index_config=Configure.inverted_index()
        )

    def set_query_arguments(self, ef):
        """
        Set query arguments for weaviate query with hnsw index
        """
        self.collection.config.update(
            vectorizer_config=Reconfigure.VectorIndex.hnsw(
                ef=ef
            )
        )
        self.name = f"WeaviateHNSW metric:{self._metric} max_connections:{self.max_connections} ef_construction:{self.ef_construction} ef:{ef}"
        print(f"[weaviate] set_query_arguments: {ef}")
        print(f"[weaviate] Collection Config: {self.collection.config.get()}")
