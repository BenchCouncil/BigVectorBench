""" Runner module for running the benchmarking. """
import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple, List, Union

import colors
import docker
import numpy as np
import psutil

from ann_benchmarks.algorithms.base.module import BaseANN

from .definitions import Definition, instantiate_algorithm
from .datasets import DATASETS, get_dataset
from .distance import dataset_transform, metrics
from .results import store_results, store_insert_update_delete_latencies


def run_individual_query(
        algo: BaseANN,
        X_train: np.array,
        X_test: np.array,
        distance: str,
        count: int,
        run_count: int,
        batch: bool,
        X_test_label : np.ndarray | None = None,
        filter_expr_func : str | None = None
        ) -> Tuple[dict, list]:
    """
    Run a search query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        X_train (np.array): The training data.
        X_test (np.array): The testing data.
        distance (str): The type of distance metric to use.
        count (int): The number of nearest neighbors to return.
        run_count (int): The number of times to run the query.
        batch (bool): Flag to indicate whether to run in batch mode or not.
        X_test_label (np.array): The labels for the testing data.
        filter_expr_func (str): The function to generate the filter expression.

    Returns:
        tuple: A tuple with the attributes of the algorithm run and the results.
    """
    num_vectors = X_train.shape[1] if len(X_train.shape) == 3 else 1
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or ((not batch) and hasattr(algo, "prepare_query"))

    assert not (batch and num_vectors > 1), "Batch mode is only supported for single-vector queries"

    best_search_time = float("inf")
    if filter_expr_func is not None:
        # assert prepared_queries == True, "Filter expression can only be used with prepared queries"
        exec(filter_expr_func, globals())
        filter_expr = globals()["filter_expr"]
    for i in range(run_count):
        print(f"Run {i + 1}/{run_count}...")
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(
                v: np.array,
                labels: Optional[np.ndarray] = None,
                ) -> Tuple[float, List[Tuple[int, float]]]:
            """
            Executes a single query on an instantiated, ANN algorithm.

            Args:
                v (np.array): Vector to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data)
            """
            expr = None
            if filter_expr_func is not None:
                expr = filter_expr(*labels)
            if prepared_queries:
                algo.prepare_query(v, count, expr)
                start = time.time()
                algo.run_prepared_query()
                total = time.time() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.time()
                candidates = algo.query(v, count, expr)
                total = time.time() - start

            # make sure all returned indices are unique
            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(metrics[distance].distance(v, X_train[idx])))
                    for idx in candidates
            ]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print(f"Processed {n_items_processed[0]}/{len(X_test)} queries...")
            if len(candidates) > count:
                print(f"warning: algorithm {algo} returned {len(candidates)} results, \
                      but count is only {count})")
            return (total, candidates)

        def batch_query(
                X: np.ndarray,
                X_labels : np.ndarray | None = None
                ) -> List[Tuple[float, List[Tuple[int, float]]]]:
            """
            Executes a batch of queries on an instantiated, ANN algorithm.

            Args:
                X (np.array): Array containing multiple vectors to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: List of tuples, each containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data)
            """
            exprs = None
            if filter_expr_func is not None:
                exprs = [filter_expr(*labels) for labels in X_labels]
            # TODO: consider using a dataclass to represent return value.
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.time()
                algo.run_prepared_batch_query()
                total = time.time() - start
            else:
                start = time.time()
                algo.batch_query(X, count, exprs)
                total = time.time() - start
            results = algo.get_batch_results()
            if hasattr(algo, "get_batch_latencies"):
                batch_latencies = algo.get_batch_latencies()
            else:
                batch_latencies = [total / float(len(X))] * len(X)

            # make sure all returned indices are unique
            for res in results:
                assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            candidates = [
                [(int(idx), float(metrics[distance].distance(v, X_train[idx])))
                 for idx in single_results]  # noqa
                for v, single_results in zip(X, results)
            ]
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        def single_multi_vector_query(vs: np.ndarray):
            """
            Executes a single query on an instantiated, ANN algorithm for multi-vector data.

            Args:
                v (np.array): Multi-vector data to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data)
            """
            algo.prepare_multi_vector_query(vs, count)
            start = time.time()
            candidates, distances = algo.run_multi_vector_query()
            total = time.time() - start
            # candidates = [
            #     # (int(idx), float(metrics[distance].distance(vs, X_train[idx])))
            #     (int(idx), float(np.sum([metrics[distance].distance(v, x) for v, x in zip(vs, X_train[idx])])))
            #     for idx in candidates
            # ]
            candidates = [(int(idx), float(d)) for idx, d in zip(candidates, distances)]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print(f"Processed {n_items_processed[0]}/{len(X_test)} queries...")
            if len(candidates) > count:
                print(f"warning: algorithm {algo} returned {len(candidates)} results, \
                      but count is only {count})")
            return (total, candidates)

        if num_vectors == 1:
            if filter_expr_func is None:
                if batch:
                    results = batch_query(X_test)
                else:
                    results = [single_query(x) for x in X_test]
            else:
                if batch:
                    results = batch_query(X_test, X_test_label)
                else:
                    results = [single_query(x, labels) for x, labels in zip(X_test, X_test_label)]
        else:
            # multi-vector ann
            if batch:
                raise NotImplementedError("Multi-vector ann datasets are not supported yet.")
            else:
                results = [single_multi_vector_query(x) for x in X_test]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def run_individual_insert(
    algo: BaseANN,
    X_test: np.array,
    X_test_label: np.ndarray | None = None
) -> list:
    """
    Run a insert query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        X_test (np.array): The testing data.
        X_test_label (np.array): The labels for the testing data.

    Returns:
        list: The latencies of the insert queries.
    """
    latencies = []
    if X_test_label is None:
        for i, x in enumerate(X_test):
            start = time.time()
            algo.insert(x)
            latencies.append(time.time() - start)
            if i % 1000 == 0:
                print(f"Processed {i}/{len(X_test)} inserts...")
    else:
        for i, (x, labels) in enumerate(zip(X_test, X_test_label)):
            start = time.time()
            algo.insert(x, labels)
            latencies.append(time.time() - start)
            if i % 1000 == 0:
                print(f"Processed {i}/{len(X_test)} inserts...")
    return latencies


def run_individual_update(
    algo: BaseANN,
    num_entities: int,
    X_test: np.array,
    X_test_label: np.ndarray | None = None
) -> list:
    """
    Run a update query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        num_entities (int): The number of entities in the database.
        X_test (np.array): The testing data.
        X_test_label (np.array): The labels for the testing data.

    Returns:
        list: The latencies of the update queries.
    """
    latencies = []
    if X_test_label is None:
        for i, x in enumerate(X_test):
            idx = np.random.randint(num_entities)
            start = time.time()
            algo.update(idx, x)
            latencies.append(time.time() - start)
            if i % 1000 == 0:
                print(f"Processed {i}/{len(X_test)} updates...")
    else:
        for i, (x, labels) in enumerate(zip(X_test, X_test_label)):
            idx = np.random.randint(num_entities)
            start = time.time()
            algo.update(idx, x, labels)
            latencies.append(time.time() - start)
            if i % 1000 == 0:
                print(f"Processed {i}/{len(X_test)} updates...")
    return latencies


def run_individual_delete(
    algo: BaseANN,
    num_entities: int,
    num_deletes: int
) -> list:
    """
    Run a delete query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        num_entities (int): The number of entities in the database.
        num_deletes (int): The number of delete queries.

    Returns:
        list: The latencies of the delete queries.
    """
    latencies = []
    delete_idxs = np.random.choice(num_entities, num_deletes, replace=False)
    for i, idx in enumerate(delete_idxs):
        start = time.time()
        algo.delete(idx)
        latencies.append(time.time() - start)
        if i % 1000 == 0:
            print(f"Processed {i}/{len(delete_idxs)} deletes...")
    return latencies


def load_and_transform_dataset(dataset_name: str) -> Tuple:
    """Loads and transforms the dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple: Transformed datasets.
    """
    D, dimension = get_dataset(dataset_name)
    if "type" in D.attrs:
        dataset_type = D.attrs["type"]
    else:
        dataset_type = "ann"
    print(f"Dataset type: {dataset_type}")
    distance = D.attrs["distance"]
    print(f"Distance metric: {distance}")
    if dataset_type == "filter-ann":
        X_train = np.array(D["train_vec"])
        X_train_label = np.array(D["train_label"])
        X_test = np.array(D["test_vec"])
        X_test_label = np.array(D["test_label"])
        label_names = D.attrs["label_names"]
        label_types = D.attrs["label_types"]
        filter_expr_func = D.attrs["filter_expr_func"]
        print(f"Got a train set of size \
              ({X_train.shape[0]} * {dimension}) with {len(X_train_label[0])} labels")
        print(f"label names: {label_names}")
        print(f"label types: {label_types}")
        print(f"filter expression function: {filter_expr_func}")
        print(f"Got {len(X_test)} queries")
        return (
            dataset_type,
            distance,
            (X_train, X_train_label, X_test, X_test_label,
             label_names, label_types, filter_expr_func),
        )
    elif dataset_type == "mv-ann":
        # multi-vector ann
        X_train = np.array(D["train"])
        X_test = np.array(D["test"])
        dimension = X_train[0].shape[1]
        print(f"Got a train set of size ({X_train.shape[0]} * {X_train.shape[1]} * {dimension})")
        print(f"Got {len(X_test)} queries")
        train, test = dataset_transform(D)
        return dataset_type, distance, (train, test)
    elif dataset_type == "mm-ann":
        # multi-modal ann
        X_train = np.array(D["train"])
        X_test = np.array(D["modal_test"])
        print(f"Got a train set of size ({X_train.shape[0]} * {dimension})")
        print(f"Got {len(X_test)} queries")
        train, test = dataset_transform(D)
        return dataset_type, distance, (train, test)
    else:
        # dataset_type = ann
        X_train = np.array(D["train"])
        X_test = np.array(D["test"])
        print(f"Got a train set of size ({X_train.shape[0]} * {dimension})")
        print(f"Got {len(X_test)} queries")
        train, test = dataset_transform(D)
        return dataset_type, distance, (train, test)


def insert_data(
    algo: BaseANN,
    X_train: np.ndarray,
    X_train_label: np.ndarray | None = None,
    label_names: List[str] | None = None,
    label_types: List[str] | None = None,
) -> Tuple:
    """
    Inserts the training data into the database of the ANN algorithm.

    Args:
        algo (BaseANN): The algorithm instance.
        X_train (np.array): The training data.
        X_train_label (np.array): The training data labels.
        label_names (List[str]): The names of the labels.
        label_types (List[str]): The types of the labels.

    Returns:
        Tuple: The insert time and memory usage.
    """
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.load_data(X_train, X_train_label, label_names, label_types)
    insert_time = time.time() - t0
    memory_usage_after = algo.get_memory_usage()
    data_size = memory_usage_after - memory_usage_before

    print("Inserted data in", insert_time)
    print("Memory usage: ", data_size)

    return insert_time, data_size


def build_index(algo: BaseANN) -> Tuple:
    """
    Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.

    Returns:
        Tuple: The build time and index size.
    """
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.create_index()
    index_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before

    print("Built index in ", index_time)
    print("Index size: ", index_size)

    return index_time, index_size

def build_multi_index(
        algo: BaseANN,
        num_vectors: int = 1
    ) -> Tuple:
    """
    Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.
        num_vectors (int): The number of vectors.

    Returns:
        Tuple: The build time and index size.
    """
    assert hasattr(algo, "create_multi_index")
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.create_multi_index(num_vectors)
    index_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before

    print("Built index in ", index_time)
    print("Index size: ", index_size)

    return index_time, index_size


def run(
        definition: Definition,
        dataset_name: str,
        count: int,
        run_count: int,
        batch: bool
        ) -> None:
    """
    Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        count (int): The number of results to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""error: query argument groups have been specified for \
        {definition.module}.{definition.constructor}({definition.arguments}), \
            but the algorithm instantiated from it does not implement \
                the set_query_arguments function"""

    dataset_type, distance, data = load_and_transform_dataset(dataset_name)
    X_train_label, X_test_label = None, None
    if dataset_type == "filter-ann":
        X_train, X_train_label, X_test, X_test_label, label_names, label_types, filter_expr_func = data
    elif dataset_type == "mv-ann":
        X_train, X_test = data[0], data[1]
    elif dataset_type == "mm-ann":
        X_train, X_test = data[0], data[1]
    else:
        # dataset_type == "ann"
        X_train, X_test = data[0], data[1]

    if hasattr(algo, "supports_prepared_queries"):
        algo.supports_prepared_queries()

    # Insert data
    if dataset_type == "filter-ann":
        insert_time, data_size = insert_data(algo, X_train, X_train_label, label_names, label_types)
    elif dataset_type == "mv-ann":
        insert_time, data_size = insert_data(algo, X_train)
    elif dataset_type == "mm-ann":
        insert_time, data_size = insert_data(algo, X_train)
    else:
        # dataset_type == "ann"
        insert_time, data_size = insert_data(algo, X_train)

    # Create or build index
    num_vectors = X_train.shape[1] if len(X_train.shape) == 3 else 1
    if dataset_type == "mv-ann":
        index_time, index_size = build_multi_index(algo, num_vectors)
    else:
        index_time, index_size = build_index(algo)

    query_argument_groups = definition.query_argument_groups or [[]]  # Ensure at least one iteration

    for pos, query_arguments in enumerate(query_argument_groups, 1):
        print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
        if query_arguments:
            algo.set_query_arguments(*query_arguments)

        if dataset_type == "ann":
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch)
        elif dataset_type == "filter-ann":
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch, X_test_label, filter_expr_func)
        elif dataset_type == "mv-ann":
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch)
        elif dataset_type == "mm-ann":
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch)
        else:
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch)
        descriptor.update({
            "insert_time": insert_time,
            "data_size": data_size,
            "index_time": index_time,
            "index_size": index_size,
            "build_time": insert_time + index_time,
            "algo": definition.algorithm,
            "dataset": dataset_name
        })
        store_results(dataset_name, count, definition, query_arguments, descriptor, results, batch)

    insert_latencies = run_individual_insert(algo, X_test, X_test_label)
    num_entities = algo.num_entities if hasattr(algo, "num_entities") else X_train.shape[0] + X_test.shape[0]
    update_latencies = run_individual_update(algo, num_entities, X_test, X_test_label)
    delete_latencies = run_individual_delete(algo, num_entities, len(X_test))
    store_insert_update_delete_latencies(dataset_name, count, definition, insert_latencies, update_latencies, delete_latencies)

    algo.done()

def run_from_cmdline():
    """
    Calls the function `run` using arguments from the command line.
    See `ArgumentParser` for arguments, all run it with `--help`.
    """
    parser = argparse.ArgumentParser(
        """
            NOTICE: You probably want to run.py rather than this script.
        """
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        help="Dataset to benchmark on.",
        required=True
    )
    parser.add_argument(
        "--algorithm",
        help="Name of algorithm for saving the results.",
        required=True
    )
    parser.add_argument(
        "--module",
        help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"',
        required=True
    )
    parser.add_argument(
        "--constructor",
        help='Constructor to load from module. E.g. "Annoy"',
        required=True
    )
    parser.add_argument(
        "-k",
        "--count",
        help="K: Number of nearest neighbours for the algorithm to return.",
        required=True,
        type=int
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorithm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument(
        "build",
        help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]'
    )
    parser.add_argument(
        "queries",
        help="JSON of arguments to pass to the queries. E.g. [100]",
        nargs="*",
        default=[]
    )
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)


def run_docker(
    definition: Definition,
    dataset: str,
    count: int,
    runs: int,
    timeout: int,
    batch: bool,
    cpuset_cpus: str,
    mem_limit: Optional[int | str] = None,
) -> None:
    """
    Runs `run_from_cmdline` within a Docker container with specified parameters
    and logs the output.

    See `run_from_cmdline` for details on the args.

    Args:
        cpuset_cpus (str): The CPUs in which to run the container.
        mem_limit (Optional[int | str]): The memory limit for the container.
    """
    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--count",
        str(count),
    ]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available

    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("/var/lib/docker/image"): {"bind": "/var/lib/docker/image", "mode": "rw"},
            os.path.abspath("/var/lib/docker/overlay2"): {"bind": "/var/lib/docker/overlay2", "mode": "rw"},
            os.path.abspath("ann_benchmarks"): {"bind": "/home/app/ann_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        cpuset_cpus=cpuset_cpus,
        mem_limit=mem_limit,
        detach=True,
        privileged=True,
        runtime="nvidia" if os.path.exists("/usr/bin/nvidia-smi") else None,
    )
    logger = logging.getLogger(f"bvb.{container.short_id}")

    logger.info(
        "Created container %s: cpuset cpus %s, mem limit %s, timeout %s, command %s",
        container.short_id,
        cpuset_cpus,
        mem_limit,
        timeout,
        cmd,
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except docker.errors.APIError as e:
        logger.error("Container.wait for container %s failed with exception", container.short_id)
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def _handle_container_return_value(
    return_value: Union[Dict[str, Union[int, str]], int],
    container: docker.models.containers.Container,
    logger: logging.Logger
) -> None:
    """
    Handles the return value of a Docker container and 
    outputs error and stdout messages (with color).

    Args:
        return_value (Union[Dict[str, Union[int, str]], int]): The return value of the container.
        container (docker.models.containers.Container): The Docker container.
        logger (logging.Logger): The logger instance.
    """

    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(return_value, dict):
        # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value.get("Error", "")
        exit_code = return_value["StatusCode"]
        msg = msg.format(f"{exit_code} with message {error_msg}")
    else:
        exit_code = return_value
        msg = msg.format(exit_code)

    if exit_code not in [0, None]:
        for line in container.logs(stream=True):
            logger.error(colors.color(line.decode(), fg="red"))
        logger.error(msg)
    else:
        logger.info(msg)
