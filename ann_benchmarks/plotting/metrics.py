""" Metrics Computations """

import numpy as np

def get_recall_values(dataset_neighbors, run_neighbors, count):
    """
    Get recall values

    :param dataset_neighbors: neighbors from the dataset
    :param run_neighbors: neighbors from the run
    :param count: number of neighbors to consider
    :return: mean, std, recall values
    """
    recalls = np.zeros(len(run_neighbors))
    for i, (true_neighbor, run_neighbor) in enumerate(zip(dataset_neighbors, run_neighbors)):
        true_neighbor = true_neighbor[:count]
        run_neighbor = run_neighbor[:count]
        tt_size = len(set(true_neighbor) & set(run_neighbor) - set([-1]))
        true_neighbor_size = len(set(true_neighbor) - set([-1]))
        recalls[i] = tt_size / float(true_neighbor_size) if true_neighbor_size > 0 else 1
    return (np.mean(recalls), np.std(recalls), recalls)


def knn(dataset_neighbors, run_neighbors, count, metrics):
    """
    Calculate the recall of the k-nn search

    :param dataset_neighbors: neighbors from the dataset
    :param run_neighbors: neighbors from the run
    :param count: number of neighbors to consider
    :param metrics: metrics group in the hdf5 file
    :return: recall values
    """
    if "knn" not in metrics:
        print("Computing knn metrics")
        knn_metrics = metrics.create_group("knn")
        mean, std, recalls = get_recall_values(dataset_neighbors, run_neighbors, count)
        knn_metrics.attrs["mean"] = mean
        knn_metrics.attrs["std"] = std
        knn_metrics["recalls"] = recalls
    else:
        print("Found cached result")
    return metrics["knn"]


def queries_per_second(times):
    """
    queries per second

    :param times: list of query times
    :return: QPS(queries per second) (1/s)
    """
    return 1 / np.mean(times)

def percentile_50(times):
    """
    50th percentile

    :param times: list of query times
    :return: 50th percentile in milliseconds
    """
    return np.percentile(times, 50.0) * 1000.0

def percentile_95(times):
    """
    95th percentile

    :param times: list of query times
    :return: 95th percentile in milliseconds
    """
    return np.percentile(times, 95.0) * 1000.0

def percentile_99(times):
    """
    99th percentile

    :param times: list of query times
    :return: 99th percentile in milliseconds
    """
    return np.percentile(times, 99.0) * 1000.0

def percentile_999(times):
    """
    99.9th percentile

    :param times: list of query times
    :return: 99.9th percentile in milliseconds
    """
    return np.percentile(times, 99.9) * 1000.0

def insert_time(attrs):
    """
    Return the insert time
    """
    return attrs["insert_time"]

def data_size(attrs):
    """
    Return the data size
    """
    return attrs["data_size"]

def index_time(attrs):
    """
    Return the time of creating the index
    """
    return attrs["index_time"]

def index_size(attrs):
    """
    Return the size of the index
    """
    return attrs.get("index_size", 0)

def build_time(attrs):
    """
    Return the time of loading data and creating the index
    """
    return attrs["build_time"]

def candidates(attrs):
    """
    Return the number of candidates generated
    """
    return attrs["candidates"]

all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: knn(
            true_neighbors, run_neighbors, run_attrs["count"], metrics
        ).attrs[
            "mean"
        ],
        "worst": float("-inf"),
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: queries_per_second(
            times
        ),
        "worst": float("-inf"),
    },
    "p50": {
        "description": "Percentile 50 (millis)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: percentile_50(times),
        "worst": float("inf"),
    },
    "p95": {
        "description": "Percentile 95 (millis)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: percentile_95(times),
        "worst": float("inf"),
    },
    "p99": {
        "description": "Percentile 99 (millis)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: percentile_99(times),
        "worst": float("inf"),
    },
    "p999": {
        "description": "Percentile 99.9 (millis)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: percentile_999(times),
        "worst": float("inf"),
    },
    "insert_time": {
        "description": "Insert time (s)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: insert_time(
            run_attrs
        ),
        "worst": float("inf"),
    },
    "data_size": {
        "description": "Data size (kB)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: data_size(
            run_attrs
        ),
        "worst": float("inf"),
    },
    "index_time": {
        "description": "Index time (s)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: index_time(
            run_attrs
        ),
        "worst": float("inf"),
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: build_time(
            run_attrs
        ),
        "worst": float("inf"),
    },
    "candidates": {
        "description": "Candidates generated",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: candidates(
            run_attrs
        ),
        "worst": float("inf"),
    },
    "index_size": {
        "description": "Index size (kB)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: index_size(
            run_attrs
        ),
        "worst": float("inf"),
    },
    "queriessize": {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda true_neighbors, run_neighbors, metrics, times, run_attrs: index_size(
            run_attrs
        )
        / queries_per_second(times),
        "worst": float("inf"),
    },
}
