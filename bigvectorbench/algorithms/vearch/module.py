""" Vearch module for BigVectorBench framework. """

import subprocess
from time import sleep, time
import numpy as np

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
    GPUIvfPQIndex,
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
        self.docker_client = None
        self.docker_name = "Vearch"
        self.container = None
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

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        num_vectors = embeddings.shape[1] if len(embeddings.shape) == 3 else 1
        num_labels = len(label_names) if label_names is not None else 0
        self.num_labels = num_labels
        self.label_names = label_names
        print(f"[Vearch] load data with {num_labels} labels!!!")
        self.client.create_database(self._database_name)
        field_vec = Field(
            name="vector",
            data_type=DataType.VECTOR,
            index=FlatIndex(
                index_name="vector_idx",
                metric_type=self._metric_type,
            ),
            dimension=self._dim,
        )
        field_id = Field(name="id", data_type=DataType.INTEGER)
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
        # print(ret.data, ret.msg)
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
            # print(ret.get_document_ids())
            # print(len(ret.get_document_ids()))
        print(f"[Vearch] Uploaded {len(embeddings)} vectors successfully!!!")
        self.num_entities = len(embeddings)

    def create_index(self):
        """Vearch has already created index during data upload"""
        pass

    def set_query_arguments(self):
        """
        Set query arguments for weaviate query with hnsw index
        """
        pass

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
            vector_infos=[
                VectorInfo("vector", self.query_vector),
            ],
            filter=self.query_filter,
            limit=self.query_topk,
        )
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
