""" Redis module for BigVectorBench framework. """

import subprocess
from time import sleep
import numpy as np

import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Num

from bigvectorbench.algorithms.base.module import BaseANN


def metric_mapping(_metric: str):
    """
    Mapping metric type to Redis distance metric

    Args:
        _metric (str): metric type

    Returns:
        str: Redis distance metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "euclidean": "L2",
        "angular": "COSINE"
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Redis] Not support metric type: {_metric}!!!")
    return _metric_type


class RedisBase(BaseANN):
    """Redis implementation"""

    def __init__(self, metric: str, dim: int):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(metric)
        self._database_name = "Redis_test"
        self.start_container()
        self.client = redis.Redis(host="localhost", port=6379)
        self.index = None
        print("[Redis] client connected successfully!!!")
        self.num_labels = 0
        self.label_names = []
        self.load_batch_size = 1000
        self.query_batch_size = 1000
        self.name = f"Redis Base metric:{self._metric}"
        self.search_params = None
        self.query = None
        self.prepare_query_results = None

    def start_container(self) -> None:
        """
        Start Redis with docker compose and wait for it to be ready
        """
        try:
            subprocess.run(
                ["docker", "compose", "down"], check=True
            )
            subprocess.run(
                ["docker", "compose", "up", "-d"], check=True
            )
            print("[Redis] docker compose up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Redis] docker compose up failed: {e}!!!")
        sleep(10)

    def stop_container(self) -> None:
        """
        Stop Redis
        """
        try:
            subprocess.run(
                ["docker", "compose", "down"], check=True
            )
            print("[Redis] docker compose down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Redis] docker compose down failed: {e}!!!")

    def get_vector_index(self):
        """Get vector index"""
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
        print(f"[Redis] load data with {num_labels} labels!!!")
        self.num_entities = len(embeddings)
        schema = {
            "index": {
                "name": "bvb_index",
                "prefix": "bvb_index_prefix",
            },
            "fields": [
                {
                    "name": "idx",
                    "type": "numeric",
                    "attrs": {"sortable": True},
                },
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": self.get_vector_index(),
                },
            ],
        }
        if num_labels > 0:
            for i in range(num_labels):
                schema["fields"].append(
                    {
                        "name": label_names[i],
                        "type": "numeric",
                        "attrs": {"sortable": True},
                    }
                )
        self.index = SearchIndex.from_dict(schema)
        self.index.set_client(self.client)
        self.index.create(overwrite=True, drop=True)
        print("[Redis] create schema successfully!!!")
        print(f"[Redis] Start uploading {len(embeddings)} vectors...")
        for i in range(0, len(embeddings), self.load_batch_size):
            items = []
            for j in range(i, min(i + self.load_batch_size, len(embeddings))):
                item = {"idx": j, "embedding": np.array(embeddings[j], dtype=np.float32).tobytes()}
                if num_labels > 0:
                    for k in range(num_labels):
                        item[label_names[k]] = int(labels[j][k])
                items.append(item)
            ret = self.index.load(items)
            # print(ret)
        print(f"[Redis] Uploaded {len(embeddings)} vectors successfully!!!")

    def create_index(self):
        """Redis has already started indexing when inserting data"""
        pass

    def create_num_filter(self, field, operator, value):
        """
        Create a Num filter

        Args:
            field: field name
            operator: operator
            value: value

        Returns:
            Num: Num filter object
        """
        if operator == ">":
            return Num(field) > value
        elif operator == "<":
            return Num(field) < value
        elif operator == ">=":
            return Num(field) >= value
        elif operator == "<=":
            return Num(field) <= value
        elif operator == "==":
            return Num(field) == value
        elif operator == "!=":
            return Num(field) != value
        else:
            raise ValueError(f"Unsupported operator for Num: {operator}")

    def convert_expr_to_filter(self, expr: str):
        """
        Convert a filter expression to a Filter object list

        Args:
            expr (str): filter expression. Example: "age > 20 and height < 180 or weight == 70"

        Returns:
            Filter: Filter object for Redis query
        """
        tokens = expr.split()
        conditons_query = []
        i = 1
        while i < len(tokens):
            if tokens[i] == "and":
                i += 2
            elif tokens[i] == "or":
                raise ValueError(
                    f"[Redis] we have not supported 'or' operator in expression!!!, expr: {expr}"
                )
            elif tokens[i] in ["==", ">=", "<=", ">", "<", "!="]:
                left = tokens[i - 1]
                operator = tokens[i]
                right = tokens[i + 1]
                i += 4
                conditons_query.append(self.create_num_filter(left, operator, int(right)))
            else:
                raise ValueError(f"[Redis] Unsupported operator: {tokens[i]}")
        combined_query = conditons_query[0]
        for i in range(1, len(conditons_query)):
            combined_query = combined_query & conditons_query[i]
        return combined_query

    def query(self, v, n, filter_expr=None):
        filters = None
        if filter_expr is not None:
            filters = self.convert_expr_to_filter(filter_expr)
        query = VectorQuery(
            vector=v.tolist(),
            vector_field_name="embedding",
            return_fields=["idx"],
            filter_expression=filters,
            num_results=n,
        )
        ret = self.index.query(query)
        return [int(item["idx"]) for item in ret]

    def prepare_query(self, v: np.array, n: int, expr: str | None = None) -> None:
        """
        Prepare query

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            expr (str): The search expression
        """
        query_filter = None
        if expr is not None:
            query_filter = self.convert_expr_to_filter(expr)
        self.query = VectorQuery(
            vector=v.tolist(),
            vector_field_name="embedding",
            return_fields=["idx"],
            filter_expression=query_filter,
            num_results=n,
        )

    def run_prepared_query(self) -> None:
        """
        Run prepared query
        """
        ret = self.index.query(self.query)
        # print(f"[Redis] query result: {ret}")
        self.prepare_query_results = [int(item["idx"]) for item in ret]

    def get_prepared_query_results(self) -> list[int]:
        """
        Get prepared query results

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        return self.prepare_query_results

    def done(self):
        self.index.delete()
        self.client.close()
        self.stop_container()

    def insert(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        Single insert data

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """
        item = {"idx": self.num_entities, "embedding": np.array(embeddings, dtype=np.float32).tobytes()}
        if self.num_labels > 0:
            for i in range(self.num_labels):
                item[self.label_names[i]] = int(labels[i])
        self.index.load([item])
        self.num_entities += 1

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
    #     print("[Redis] Not support update data!!!")

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
    #     print("[Redis] Not support delete data!!!")


class RedisFlat(RedisBase):
    """Redis Flat implementation"""

    def __init__(self, metric: str, dim: int):
        super().__init__(metric, dim)
        self.name = f"Redis Flat metric:{self._metric}"

    def get_vector_index(self):
        """Get Flat vector index"""
        return {
            "dims": self._dim,
            "distance_metric": self._metric_type,
            "algorithm": "flat"
        }


class RedisHNSW(RedisBase):
    """Redis HNSW implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._m = index_param.get("m", 16)
        self._ef_construction = index_param.get("ef_construction", 200)
        self._ef_runtime = index_param.get("ef_runtime", 10)
        self._epsilon = index_param.get("epsilon", 0.01)

    def get_vector_index(self):
        """Get HNSW vector index"""
        return {
            "dims": self._dim,
            "distance_metric": self._metric_type,
            "algorithm": "hnsw",
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_runtime": self._ef_runtime,
            "epsilon": self._epsilon
        }
