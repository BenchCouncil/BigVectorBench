""" Elasticsearch module for BigVectorBench framework. """

import subprocess
from time import sleep
import numpy as np

from elasticsearch import Elasticsearch

from bigvectorbench.algorithms.base.module import BaseANN

def metric_mapping(_metric: str):
    """
    Mapping metric type to Elasticsearch distance metric

    Args:
        _metric (str): metric type

    Returns:
        str: Elasticsearch distance metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "angular": "cosine",
        "euclidean": "l2_norm",
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Elasticsearch] Not support metric type: {_metric}!!!")
    return _metric_type

class ElasticsearchBase(BaseANN):
    """Elasticsearch implementation"""
    def __init__(self, metric: str, dim: int):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(metric)
        self._database_name = "Elasticsearch_test"

        self.start_container()

        self.client = Elasticsearch("http://localhost:9200")
        if self.client.ping():
            print("[Elasticsearch] client connected successfully!!!")
        else:
            print("[Elasticsearch] client connected failed!!!")

        self.num_labels = 0
        self.label_names = []
        self.load_batch_size = 1000
        self.query_batch_size = 100

        self.name = f"Elasticsearch Base metric:{self._metric}"
        self.search_params = None

        self.index_name = ""
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
        Start Elasticsearch with docker compose and wait for it to be ready
        """
        try:
            subprocess.run(
                ["docker", "compose", "down"], check=True
            )
            subprocess.run(
                ["docker", "compose",  "up", "-d"], check=True
            )
            print("[Elasticsearch docker compose up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Elasticsearch] docker compose up failed: {e}!!!")
        sleep(30)

    def stop_container(self) -> None:
        """
        Stop Elasticsearch
        """
        try:
            subprocess.run(
                ["docker", "compose", "down"], check=True
            )
            print("[Elasticsearch] docker compose down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Elasticsearch] docker compose down failed: {e}!!!")

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
        print(f"[Elasticsearch] load data with {num_labels} labels!!!")
        self.num_entities = len(embeddings)
        self.index_name = "bvb_index"
        settings ={
            "index" : {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": -1,
            }
        }
        mappings = {
            "properties": {
                "id": {
                    "type": "keyword", 
                    "store": True
                },
                "embedding": {
                    "type": "dense_vector", 
                    "dims": self._dim, 
                    "index": True, 
                    "similarity": self._metric_type, 
                    # "index_options": self.get_vector_index()
                }
            }
        }
        if num_labels > 0:
            for i, label_name in enumerate(label_names):
                mappings["properties"][label_name] = {"type": "keyword","store": True,}

        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body={"settings": settings, "mappings": mappings})

        print("[Elasticsearch] create index successfully!!!")
        print(f"[Elasticsearch] Start uploading {len(embeddings)} vectors...")

        for i in range(0, len(embeddings), self.load_batch_size):
            items = []
            for j in range(i, min(i +  self.load_batch_size, len(embeddings))):
                items.append({"index": {"_index": self.index_name}})
                item = {"id": j, "embedding": embeddings[j].tolist()}
                if num_labels > 0:
                    for k in range(num_labels):
                        item[label_names[k]] = int(labels[j][k])
                items.append(item)
            # use Elasticsearch bulk API to upload batch data
            ret = self.client.bulk(operations=items, index=self.index_name)
            # refreshing index to make it can be found
            self.client.indices.refresh(index=self.index_name)
        print(f"[Elasticsearch] Uploaded {len(embeddings)} vectors successfully!!!")

        # print("Force merge index ...")
        # self.client.indices.forcemerge(index=self.index_name, max_num_segments=1, request_timeout=900)

        # print("Refreshing index ...")
        # self.client.indices.refresh(index=self.index_name, request_timeout=900)

    def create_index(self):
        """Elasticsearch has already started indexing when inserting data"""
        pass

    def set_query_arguments(self):
        """
        Set query arguments for elasticsearch query with hnsw index
        """
        raise NotImplementedError()


    def convert_expr_to_filter(self, expr: str):
        """
        Convert a filter expression to a Filter object list

        Args:
            expr (str): filter expression. Example: "age > 20 and height < 180 or weight == 70"

        Returns:
            Filter: Filter object for Elasticsearch query
        """
        tokens = expr.split()
        must_filters = []
        must_not_filters = []

        i = 1
        while i < len(tokens):
            if tokens[i] == "and":
                i += 2
            elif tokens[i] == "or":
                raise ValueError(
                    f"[Elasticsearch] we have not supported 'or' operator in expression!!!, expr: {expr}"
                )
            elif tokens[i] in ["==", ">=", "<=", ">", "<", "!="]:
                left = tokens[i - 1]
                operator = tokens[i]
                right = tokens[i + 1]
                i += 4

                # range, gte >=, lte <=, ,gt >, lt <
                # term,=
                if operator == ">=":
                    must_filters.append(
                        {"range":{left : { "gte" : int(right)}}}
                    )
                elif operator == "<=":
                    must_filters.append(
                        {"range":{left : { "lte" : int(right)}}}
                    )
                elif operator == ">":
                    must_filters.append(
                        {"range":{left : { "gt" : int(right)}}}
                    )
                elif operator == "<":
                    must_filters.append(
                        {"range":{left : { "lt" : int(right)}}}
                    )
                elif operator == "==":
                    must_filters.append(
                        {"term":{left : int(right)}}
                    )
                elif operator == "!=":
                    must_not_filters.append(
                        {"term":{left : int(right)}}
                    )
            else:
                raise ValueError(f"[Elasticsearch] Unsupported operator: {tokens[i]}")
        return must_filters, must_not_filters

    def query(self, v, n, filter_expr=None):
        if filter_expr is not None:
            must_filters, must_not_filters = self.convert_expr_to_filter(filter_expr)
        else:
            must_filters = []
            must_not_filters = []

        body = {
            "knn": {
                "field": "embedding",
                "query_vector": v.tolist(),
                "k": n,
                "num_candidates": self.query_batch_size,
                "filter": {
                    "bool" : {
                        "must" : must_filters,
                        "must_not" : must_not_filters,
                    }
                },  
            },
        }
        res = self.client.search(
            index=self.index_name,
            body=body,
            size=n,
            # _source=False,
            # docvalue_fields=["id"],
            # # stored_fields="_none_",
            # filter_path=["hits.hits._source.id"],
            request_timeout=10,
        )
        return [int(h["_source"]["id"]) for h in res["hits"]["hits"]]

    # def prepare_query(self, v: np.array, n: int, expr: str | None = None) -> None:
    #     """
    #     Prepare query

    #     Args:
    #         v (np.array): The vector to find the nearest neighbors of.
    #         n (int): The number of nearest neighbors to return.
    #         expr (str): The search expression
    #     """
    #     self.query_vector = v.tolist()
    #     self.query_topk = n
    #     if expr is not None:
    #         self.query_filter = self.convert_expr_to_filter(expr)

    # def run_prepared_query(self) -> None:
    #     """
    #     Run prepared query
    #     """
    #     ret = self.client.search(
    #         database_name=self._database_name,
    #         space_name=f"{self._database_name}_space",
    #         index_params=self.search_params,
    #         vector_infos=[
    #             VectorInfo("vector", self.query_vector),
    #         ],
    #         filter=self.query_filter,
    #         limit=self.query_topk,
    #     )
    #     # print(f"[Elasticsearch] query result: {ret.__dict__}")
    #     self.prepare_query_results = [point["id"] for point in ret.documents[0]]

    # def get_prepared_query_results(self) -> list[int]:
    #     """
    #     Get prepared query results

    #     Returns:
    #         list[int]: An array of indices representing the nearest neighbors.
    #     """
    #     return self.prepare_query_results

    def done(self):
        self.client.indices.delete(index=self.index_name)
        self.client.close()
        print("[Elasticsearch] collection deleted successfully!!!")
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
        items = []
        items.append({"index": {"_index": self.index_name}})
        item = {"id": self.num_entities + 1, "embedding": embeddings.tolist()}

        if self.num_labels > 0:
            assert len(labels) == self.num_labels
            for i in range(self.num_labels):
                item[self.label_names[i]] = labels[i]
        
        items.append(item)
        ret = self.client.bulk(operations=items, index=self.index_name)
        self.client.indices.refresh(index=self.index_name)
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
    #     item = {"id": index, "vector": embeddings.tolist()}
    #     if self.num_labels > 0:
    #         assert len(labels) == self.num_labels
    #         for i in range(self.num_labels):
    #             item[self.label_names[i]] = int(labels[i])
    #     self.client.upsert(
    #         database_name=self._database_name,
    #         space_name=f"{self._database_name}_space",
    #         data=[item],
    #     )

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
    #     # print(f"[Elasticsearch] delete index: {index}")
    #     self.client.delete(
    #         database_name=self._database_name,
    #         space_name=f"{self._database_name}_space",
    #         filter=Filter(
    #             operator="AND",
    #             conditions=[
    #                 Condition(operator="=", fv=FieldValue(field="id", value=int(index)))
    #             ],
    #         ),
    #         limit=1,
    #     )
    
class ElasticsearchHNSW(ElasticsearchBase):
    """Elasticsearch HNSW implementation"""

    def __init__(self, metric: str, dim: int, index_param: dict):
        super().__init__(metric, dim)
        self._nlinks = index_param.get("nlinks", 32)
        self._efConstruction = index_param.get("efConstruction", 40)

    def get_vector_index(self):
        """Get HNSW vector index"""
        return {
            "index_name":"vector_idx",
            "type": "hnsw",
            "metric_type":self._metric_type,
            "nlinks":self._nlinks,
            "efConstruction":self._efConstruction,
        }

    def set_query_arguments(self, efSearch: int = 40):
        """
        Set query arguments for Elasticsearch query with hnsw index
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "efSearch": efSearch,
        }
        self.name = f"Elasticsearch HNSW metric:{self._metric}, nlinks:{self._nlinks}, efConstruction:{self._efConstruction}, efSearch:{efSearch}"
