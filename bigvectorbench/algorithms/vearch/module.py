""" Vearch module for BigVectorBench framework. """

import subprocess
from time import sleep
import numpy as np
import requests

from vearch.core.vearch import Vearch
from vearch.config import Config
from vearch.schema.field import Field
from vearch.schema.space import SpaceSchema
from vearch.utils import DataType, MetricType, VectorInfo
from vearch.schema.index import (
    FlatIndex,
    ScalarIndex,
    HNSWIndex,
    BinaryIvfIndex,
    IvfFlatIndex,
    IvfPQIndex,
)
from vearch.filter import Filter, Condition, FieldValue

from bigvectorbench.algorithms.base.module import BaseANN


def metric_mapping(_metric: str):
    """
    Mapping metric type to Vearch distance metric

    Args:
        _metric (str): metric type

    Returns:
        str: Vearch distance metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "euclidean": MetricType.L2,
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Vearch] Not support metric type: {_metric}!!!")
    return _metric_type


class VearchBase(BaseANN):
    """Vearch implementation"""

    def __init__(self, metric: str, dim: int):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(metric)
        self._database_name = "Vearch_test"
        self.start_container()
        config = Config(host="http://localhost:9001", token="secret")
        self.client = Vearch(config)
        print("[Vearch] client connected successfully!!!")
        self.num_labels = 0
        self.label_names = []
        self.load_batch_size = 1000
        self.query_batch_size = 100
        if self.client.is_database_exist(self._database_name):
            print("[Vearch] collection already exists!!!")
            self.client.drop_database(self._database_name)
            print("[Vearch] collection deleted successfully!!!")
        self.name = f"Vearch Base metric:{self._metric}"
        self.search_params = None
        self.query_vector = None
        self.query_topk = 0
        self.query_filter = None
        self.prepare_query_results = None
        self.batch_search_queries = []
        self.batch_results = []
        self.batch_latencies = []
        super().__init__()

    def start_container(self) -> None:
        """
        Start Vearch with docker compose and wait for it to be ready
        """
        try:
            subprocess.run(
                ["docker", "compose", "--profile", "standalone", "down"], check=True
            )
            subprocess.run(
                ["docker", "compose", "--profile", "standalone", "up", "-d"], check=True
            )
            print("[Vearch] docker compose --profile standalone up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Vearch] docker compose --profile standalone up failed: {e}!!!")
        sleep(10)

    def stop_container(self) -> None:
        """
        Stop Vearch
        """
        try:
            subprocess.run(
                ["docker", "compose", "--profile", "standalone", "down"], check=True
            )
            print("[Vearch] docker compose --profile standalone down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Vearch] docker compose --profile standalone down failed: {e}!!!")

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
        print(f"[Vearch] load data with {num_labels} labels!!!")
        ret = self.client.create_database(self._database_name)
        print(f"[Vearch] create database: {ret.__dict__}")
        # number of entities, needed for training_threshold in IvfFlatIndex and IvfPQIndex
        self.num_entities = len(embeddings)
        field_vec = Field(
            name="vector",
            data_type=DataType.VECTOR,
            index=self.get_vector_index(),
            dimension=self._dim,
        )
        field_id = Field(name="id", data_type=DataType.INTEGER, index=ScalarIndex(index_name="id_idx"))
        fields = [field_id, field_vec]
        for i in range(num_labels):
            field = Field(
                name=label_names[i],
                data_type=DataType.INTEGER,
                index=ScalarIndex(index_name=f"{label_names[i]}_idx"),
            )
            fields.append(field)
        space_schema = SpaceSchema(name=f"{self._database_name}_space", fields=fields)
        ret = self.client.create_space(
            database_name=self._database_name, space=space_schema
        )
        print(f"[Vearch] create space: {ret.__dict__}")
        print("[Vearch] database and space created successfully!!!")
        print(f"[Vearch] Start uploading {len(embeddings)} vectors...")
        for i in range(0, len(embeddings), self.load_batch_size):
            items = []
            for j in range(i, min(i + self.load_batch_size, len(embeddings))):
                item = {"id": j, "vector": embeddings[j].tolist()}
                if num_labels > 0:
                    for k in range(num_labels):
                        item[label_names[k]] = int(labels[j][k])
                items.append(item)
            ret = self.client.upsert(
                database_name=self._database_name,
                space_name=f"{self._database_name}_space",
                data=items,
            )
            # print(ret.__dict__)
            # print(ret.get_document_ids())
            # print(len(ret.get_document_ids()))
        print(f"[Vearch] Uploaded {len(embeddings)} vectors successfully!!!")

    def waiting_index_finish(self, total, timewait=5):
        """'Waiting for index finish'"""
        router_url = "http://localhost:9001"
        url = (
            router_url
            + "/dbs/"
            + self._database_name
            + "/spaces/"
            + self._database_name
            + "_space"
        )
        num = 0
        while num < total:
            num = 0
            response = requests.get(url, auth=("root", "secret"), timeout=10)
            # print(f"response: {response.json()}")
            partitions = response.json()["data"]["partitions"]
            for p in partitions:
                num += p["index_num"]
            # print(f"index num: {num}/{total}")
            sleep(timewait)

    def create_index(self):
        """Vearch has already started indexing when inserting data"""
        print("[Vearch] Waiting for index finish...")
        self.waiting_index_finish(self.num_entities, timewait=1)
        print("[Vearch] Index finish!!!")

    def set_query_arguments(self):
        """
        Set query arguments for weaviate query with hnsw index
        """
        raise NotImplementedError()

    def convert_expr_to_filter(self, expr: str):
        """
        Convert a filter expression to a Filter object list

        Args:
            expr (str): filter expression. Example: "age > 20 and height < 180 or weight == 70"

        Returns:
            Filter: Filter object for Vearch query
        """
        tokens = expr.split()
        conditons_query = []
        i = 1
        while i < len(tokens):
            if tokens[i] == "and":
                i += 2
            elif tokens[i] == "or":
                raise ValueError(
                    f"[Vearch] we have not supported 'or' operator in expression!!!, expr: {expr}"
                )
            elif tokens[i] in ["==", ">=", "<=", ">", "<", "!="]:
                left = tokens[i - 1]
                operator = tokens[i]
                right = tokens[i + 1]
                i += 4
                fv = FieldValue(field=left, value=int(right))
                if operator == "==": # Vearch seems not support "=" operator
                    conditons_query.append(Condition(operator=">=", fv=fv))
                    conditons_query.append(Condition(operator="<=", fv=fv))
                else:
                    conditons_query.append(Condition(operator=operator, fv=fv))
            else:
                raise ValueError(f"[Vearch] Unsupported operator: {tokens[i]}")
        return Filter(operator="AND", conditions=conditons_query)

    def query(self, v, n, filter_expr=None):
        filters = None
        if filter_expr is not None:
            filters = self.convert_expr_to_filter(filter_expr)
        vi = VectorInfo("vector", v)
        ret = self.client.search(
            database_name=self._database_name,
            space_name=f"{self._database_name}_space",
            index_params=self.search_params,
            vector_infos=[vi],
            filter=filters,
            limit=n,
        )
        return [point["id"] for point in ret.documents[0]]

    def prepare_query(self, v: np.array, n: int, expr: str | None = None) -> None:
        """
        Prepare query

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            expr (str): The search expression
        """
        self.query_vector = v.tolist()
        self.query_topk = n
        if expr is not None:
            self.query_filter = self.convert_expr_to_filter(expr)

    def run_prepared_query(self) -> None:
        """
        Run prepared query
        """
        ret = self.client.search(
            database_name=self._database_name,
            space_name=f"{self._database_name}_space",
            index_params=self.search_params,
            vector_infos=[
                VectorInfo("vector", self.query_vector),
            ],
            filter=self.query_filter,
            limit=self.query_topk,
        )
        # print(f"[Vearch] query result: {ret.__dict__}")
        self.prepare_query_results = [point["id"] for point in ret.documents[0]]

    def get_prepared_query_results(self) -> list[int]:
        """
        Get prepared query results

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        return self.prepare_query_results

    def done(self):
        self.client.drop_space(
            database_name=self._database_name,
            space_name=f"{self._database_name}_space",
        )
        self.client.drop_database(self._database_name)
        print("[Vearch] collection deleted successfully!!!")
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
        item = {"id": self.num_entities + 1, "vector": embeddings.tolist()}
        if self.num_labels > 0:
            assert len(labels) == self.num_labels
            for i in range(self.num_labels):
                item[self.label_names[i]] = int(labels[i])
        self.client.upsert(
            database_name=self._database_name,
            space_name=f"{self._database_name}_space",
            data=[item],
        )
        self.num_entities += 1

    def update(
        self, index: int, embeddings: np.ndarray, labels: np.ndarray | None = None
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
        item = {"id": index, "vector": embeddings.tolist()}
        if self.num_labels > 0:
            assert len(labels) == self.num_labels
            for i in range(self.num_labels):
                item[self.label_names[i]] = int(labels[i])
        self.client.upsert(
            database_name=self._database_name,
            space_name=f"{self._database_name}_space",
            data=[item],
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
        # print(f"[Vearch] delete index: {index}")
        self.client.delete(
            database_name=self._database_name,
            space_name=f"{self._database_name}_space",
            filter=Filter(
                operator="AND",
                conditions=[
                    Condition(operator="=", fv=FieldValue(field="id", value=int(index)))
                ],
            ),
            limit=1,
        )


class VearchFlat(VearchBase):
    """Vearch Flat implementation"""

    def __init__(self, metric: str, dim: int):
        super().__init__(metric, dim)
        self.name = f"Vearch Flat metric:{self._metric}"

    def get_vector_index(self):
        """Get Flat vector index"""
        return FlatIndex(
            index_name="vector_idx",
            metric_type=self._metric_type,
        )

    def set_query_arguments(self):
        """
        Set query arguments for vearch query with flat index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "parallel_on_queries": 0,
        }


class VearchHNSW(VearchBase):
    """Vearch HNSW implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._nlinks = index_param.get("nlinks", 32)
        self._efConstruction = index_param.get("efConstruction", 40)

    def get_vector_index(self):
        """Get HNSW vector index"""
        return HNSWIndex(
            index_name="vector_idx",
            metric_type=self._metric_type,
            nlinks=self._nlinks,
            efConstruction=self._efConstruction,
        )

    def set_query_arguments(self, efSearch: int = 40):
        """
        Set query arguments for veaviate query with hnsw index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "efSearch": efSearch,
        }
        self.name = f"Vearch HNSW metric:{self._metric}, nlinks:{self._nlinks}, efConstruction:{self._efConstruction}, efSearch:{efSearch}"


class VearchIvfFlat(VearchBase):
    """Vearch IvfFlat implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._ncentroids = index_param.get("ncentroids", 2048)

    def get_vector_index(self):
        """Get IvfFlat vector index"""
        return IvfFlatIndex(
            index_name="vector_idx",
            metric_type=self._metric_type,
            ncentroids=self._ncentroids,
            training_threshold=self.num_entities,
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
        self.name = f"Vearch IvfFlat metric:{self._metric}, ncenroids:{self._ncentroids}, nprobe:{nprobe}"


class VearchIvfPQ(VearchBase):
    """Vearch IvfPQ implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._ncentroids = index_param.get("ncentroids", 2048)
        self._nsubvector = index_param.get("nsubvector", 64)

    def get_vector_index(self):
        """Get IvfPQ vector index"""
        return IvfPQIndex(
            index_name="vector_idx",
            metric_type=self._metric_type,
            ncentroids=self._ncentroids,
            nsubvector=self._nsubvector,
            training_threshold=self.num_entities,
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
        self.name = f"Vearch IvfPQ metric:{self._metric}, ncenroids:{self._ncentroids}, nsuvectors:{self._nsubvector}, nprobe:{nprobe}"


class VearchBinaryIvf(VearchBase):
    """Vearch BinaryIvf implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._ncentroids = index_param.get("ncentroids", 2048)

    def get_vector_index(self):
        """Get BinaryIvf vector index"""
        return BinaryIvfIndex(
            index_name="vector_idx",
            metric_type=self._metric_type,
            ncentroids=self._ncentroids,
        )

    def set_query_arguments(self, nprobe: int = 80):
        """
        Set query arguments for weaviate query with hnsw index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "nprobe": nprobe,
        }
        self.name = f"Vearch BinaryIvf metric:{self._metric}, ncenroids:{self._ncentroids}, nprobe:{nprobe}"
