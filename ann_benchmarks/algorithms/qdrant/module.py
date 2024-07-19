""" Qdrant module for ANN_Benchmarks framework. """
import subprocess
from time import sleep, time
import docker
import numpy as np

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition
from qdrant_client.http.models import PointStruct, CollectionStatus

from ann_benchmarks.algorithms.base.module import BaseANN

def metric_mapping(_metric: str):
    """
    Mapping metric type to Qdrant distance metric

    Args:
        _metric (str): metric type

    Returns:
        str: Qdrant distance metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "dot": Distance.DOT,
        "angular": Distance.COSINE,
        "euclidean": Distance.EUCLID
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Qdrant] Not support metric type: {_metric}!!!")
    return _metric_type


class Qdrant(BaseANN):
    """ Qdrant implementation """
    def __init__(
            self,
            metric : str,
            index_param : dict
        ):
        self._metric = metric
        self._metric_type = metric_mapping(metric)
        self._collection_name = "qdrant_test"
        self._m = index_param.get("M", None)
        self._ef_construct = index_param.get("efConstruction", None)
        self.docker_client = None
        self.docker_name = "qdrant"
        self.container = None
        self.start_container()
        self.client = QdrantClient(url="http://localhost:6333", timeout=10)
        print("[qdrant] client connected successfully!!!")
        self.num_labels = 0
        self.label_names = []
        self.load_batch_size = 1000
        self.query_batch_size = 100
        if self.client.collection_exists(self._collection_name):
            print("[qdrant] collection already exists!!!")
            self.client.delete_collection(self._collection_name)
            print("[qdrant] collection deleted successfully!!!")
        self.name = f"Qdrant metric:{self._metric} m:{self._m} ef_construct:{self._ef_construct}"
        self.search_params = None
        self.query_vector = None
        self.query_topk = 0
        self.query_filter_must = None
        self.query_filter_must_not = None
        self.prepare_query_results = None
        self.batch_search_queries = []
        self.batch_results = []
        self.batch_latencies = []

    def start_container(self) -> None:
        """
        Start qdrant by docker run
        """
        self.docker_client = docker.from_env()
        if self.docker_client.containers.list(filters={"name": self.docker_name}) != []:
            print("[qdrant] docker container already exists!!!")
            self.container = self.docker_client.containers.get(self.docker_name)
            self.stop_container()
        subprocess.run(["docker", "pull", "qdrant/qdrant:v1.9.2"], check=True)
        self.container = self.docker_client.containers.run(
            "qdrant/qdrant:v1.9.2",
            name=self.docker_name,
            volumes={
                "/tmp/qdrant_storage": {
                    "bind": "/qdrant/storage",
                    "mode": "z"
                }
            },
            ports={
                "6333/tcp": 6333,
                "6334/tcp": 6334
            },
            detach=True
        )
        print("[qdrant] docker start successfully!!!")
        sleep(10)

    def stop_container(self) -> None:
        """
        Stop qdrant
        """
        self.container.stop()
        self.container.remove(force=True)
        print("[qdrant] docker stop successfully!!!")

    def load_data(
            self,
            embeddings: np.array,
            labels: np.ndarray | None = None,
            label_names: list[str] | None = None,
            label_types: list[str] | None = None,
            ) -> None:
        dimensions = embeddings.shape[1]
        num_labels = len(label_names) if label_names is not None else 0
        self.num_labels = num_labels
        self.label_names = label_names
        print(f"[qdrant] load data with {num_labels} labels!!!")
        self.client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=self._metric_type,
                on_disk=True
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
            on_disk_payload=True
        )
        print("[qdrant] collection created successfully!!!")
        print(f"[qdrant] Start uploading {len(embeddings)} vectors...")
        for i in range(0, len(embeddings), self.load_batch_size):
            points = []
            for j in range(i, min(i + self.load_batch_size, len(embeddings))):
                payload = {}
                if num_labels > 0:
                    for k in range(num_labels):
                        payload[label_names[k]] = int(labels[j][k])
                points.append(PointStruct(
                    id=j,
                    vector=embeddings[j],
                    payload=payload
                ))
            self.client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True
            )
        print(f"[qdrant] Uploaded {len(embeddings)} vectors successfully!!!")
        # wait for vectors to be fully indexed
        while True:
            sleep(5)
            collection_info = self.client.get_collection(self._collection_name)
            if collection_info.status != CollectionStatus.GREEN:
                continue
            else:
                print(f"[qdrant] Point count: {collection_info.points_count}")
                print(f"[qdrant] Stored vectors: {collection_info.vectors_count}")
                print(f"[qdrant] Indexed vectors: {collection_info.indexed_vectors_count}")
                print(f"[qdrant] Collection status: {collection_info.status}")
                break
        self.num_entities = len(embeddings)

    def create_index(self):
        """ Qdrant has already created index during data upload """
        self.client.update_collection(
            collection_name=self._collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=self._m,
                ef_construct=self._ef_construct,
            ),
            quantization_config=models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=models.CompressionRatio.X32,
                    always_ram=True
                )
            ),
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

    def set_query_arguments(self, ef, exact):
        """
        Set query arguments for weaviate query with hnsw index
        """
        self.search_params = models.SearchParams(hnsw_ef=ef, exact=exact)
        self.name = f"Qdrant metric:{self._metric} m:{self._m} ef_construct:{self._ef_construct} ef:{ef} exact:{exact}"

    def convert_expr_to_filter(self, expr: str):
        """
        Convert a filter expression to a Filter object list

        Args:
            expr (str): filter expression. Example: "age > 20 and height < 180 or weight == 70"

        Returns:
            Filter: Filter object for qdrant query
        """
        tokens = expr.split()
        must_filters = []
        must_not_filters = []

        i = 1
        while i < len(tokens):
            if tokens[i] == "and":
                i += 2
            elif tokens[i] == "or":
                raise ValueError(f"[qdrant] we have not supported 'or' operator in expression!!!, expr: {expr}")
            elif tokens[i] in ["==", ">=", "<=", ">", "<", "!="]:
                left = tokens[i - 1]
                operator = tokens[i]
                right = tokens[i + 1]
                i += 4
                if operator == ">=":
                    must_filters.append(FieldCondition(key=left, range=models.Range(gte=int(right))))
                elif operator == "<=":
                    must_filters.append(FieldCondition(key=left, range=models.Range(lte=int(right))))
                elif operator == ">":
                    must_filters.append(FieldCondition(key=left, range=models.Range(gt=int(right))))
                elif operator == "<":
                    must_filters.append(FieldCondition(key=left, range=models.Range(lt=int(right))))
                elif operator == "==":
                    must_filters.append(FieldCondition(key=left, match=models.MatchValue(value=int(right))))
                elif operator == "!=":
                    must_not_filters.append(FieldCondition(key=left, match=models.MatchValue(value=int(right))))
            else:
                raise ValueError(f"[qdrant] Unsupported operator: {tokens[i]}")
        return must_filters, must_not_filters

    def query(self, v, n, filter_expr=None):
        if filter_expr is not None:
            must_filters, must_not_filters = self.convert_expr_to_filter(filter_expr)
        else:
            must_filters = []
            must_not_filters = []
        ret = self.client.search(
            collection_name=self._collection_name,
            query_vector=v,
            query_filter=Filter(
                must = must_filters,
                must_not = must_not_filters
            ),
            search_params=self.search_params,
            limit=n,
        )
        return [point.id for point in ret]

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
        if expr is not None:
            self.query_filter_must, self.query_filter_must_not = self.convert_expr_to_filter(expr)
        else:
            self.query_filter_must, self.query_filter_must_not = None, None

    def run_prepared_query(self) -> None:
        """
        Run prepared query
        """
        ret = self.client.search(
            collection_name=self._collection_name,
            query_vector=self.query_vector,
            query_filter=Filter(
                must = self.query_filter_must,
                must_not = self.query_filter_must_not
            ),
            search_params=self.search_params,
            limit=self.query_topk,
        )
        self.prepare_query_results = [point.id for point in ret]

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
        if exprs is not None:
            batch_query_filters_must = []
            batch_query_filters_must_not = []
            for expr in exprs:
                must_filters, must_not_filters = self.convert_expr_to_filter(expr)
                batch_query_filters_must.append(must_filters)
                batch_query_filters_must_not.append(must_not_filters)
        else:
            batch_query_filters_must = None
            batch_query_filters_must_not = None
        for i in range(0, len(vectors), self.query_batch_size):
            search_queries = []
            for j in range(i, min(i + self.query_batch_size, len(vectors))):
                search_queries.append(
                    models.SearchRequest(
                        vector=vectors[j],
                        filter=Filter(
                            must=batch_query_filters_must[j]
                                if batch_query_filters_must is not None
                                else [],
                            must_not=
                                batch_query_filters_must_not[j]
                                if batch_query_filters_must_not is not None
                                else [],
                        ),
                        params=self.search_params,
                        limit=n
                    )
                )
            self.batch_search_queries.append(search_queries)

    def run_prepared_batch_query(self) -> None:
        """
        Run prepared batch query
        """
        for search_queries in self.batch_search_queries:
            start = time()
            ret = self.client.search_batch(
                collection_name=self._collection_name,
                requests=search_queries
            )
            end = time()
            self.batch_latencies.extend([(end - start) / len(search_queries)] * len(search_queries))
            for result in ret:
                self.batch_results.append([point.id for point in result])

    def get_batch_results(self) -> list[list[int]]:
        """
        Get batch query results

        Returns:
            list[list[int]]: An array of arrays of indices representing the nearest neighbors.
        """
        return self.batch_results

    def get_batch_latencies(self) -> list[float]:
        """
        Get batch query latencies

        Returns:
            list[float]: An array of latencies for each query.
        """
        return self.batch_latencies

    def done(self):
        self.client.delete_collection(self._collection_name)
        print("[qdrant] collection deleted successfully!!!")
        self.client.close()
        self.stop_container()

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
        payload = {}
        if self.num_labels > 0:
            assert len(labels) == self.num_labels
            for i in range(self.num_labels):
                payload[self.label_names[i]] = int(labels[i])
        point = PointStruct(
            id=self.num_entities,
            vector=embeddings,
            payload=payload
        )
        self.client.upsert(
            collection_name=self._collection_name,
            points=[point],
            wait=True
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
        payload = {}
        if self.num_labels > 0:
            assert len(labels) == self.num_labels
            for i in range(self.num_labels):
                payload[self.label_names[i]] = int(labels[i])
        point = PointStruct(
            id=index,
            vector=embeddings,
            payload=payload
        )
        self.client.upsert(
            collection_name=self._collection_name,
            points=[point],
            wait=True
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
        # print(f"[qdrant] delete index: {index}")
        self.client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(
                points=[int(index)]
            ),
            wait=True
        )
