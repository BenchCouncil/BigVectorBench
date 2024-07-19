""" This module provides functions for storing and loading results from HDF5 files. """
import json
import os
import re
import traceback
from typing import Any, Optional, Set, Tuple, Iterator
import h5py

from ann_benchmarks.definitions import Definition


def build_result_filepath(dataset_name: Optional[str] = None, 
                          count: Optional[int] = None, 
                          definition: Optional[Definition] = None, 
                          query_arguments: Optional[Any] = None, 
                          batch_mode: bool = False) -> str:
    """
    Constructs the filepath for storing the results.

    Args:
        dataset_name (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        definition (Definition, optional): The definition of the algorithm.
        query_arguments (Any, optional): Additional arguments for the query.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Returns:
        str: The constructed filepath.
    """
    d = ["results"]
    if dataset_name:
        d.append(dataset_name)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm + ("-batch" if batch_mode else ""))
        data = definition.arguments + query_arguments
        d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5")
    return os.path.join(*d)


def store_results(
    dataset_name: str,
    count: int,
    definition: Definition,
    query_arguments: Any,
    attrs: dict,
    results: list,
    batch: bool = False
):
    """
    Stores results for an algorithm (and hyperparameters) running against a dataset in a HDF5 file.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition): The definition of the algorithm.
        query_arguments (Any): Additional arguments for the query.
        attrs (dict): Attributes to be stored in the file.
        results (list): Results to be stored.
        batch (bool): If True, the batch mode is activated.
    """
    filename = build_result_filepath(dataset_name, count, definition, query_arguments, batch)
    directory, _ = os.path.split(filename)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(filename, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        times = f.create_dataset("times", (len(results),), "f")
        neighbors = f.create_dataset("neighbors", (len(results), count), "i")
        distances = f.create_dataset("distances", (len(results), count), "f")

        for i, (time, ds) in enumerate(results):
            times[i] = time
            neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
            distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))


def build_latencies_filepath(
    dataset_name: Optional[str] = None, 
    count: Optional[int] = None, 
    definition: Optional[Definition] = None, 
    batch_mode: bool = False
) -> str:
    """
    Constructs the filepath for storing the latencies.

    Args:
        dataset_name (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        definition (Definition, optional): The definition of the algorithm.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Returns:
        str: The constructed filepath.
    """
    d = ["results"]
    if dataset_name:
        d.append(dataset_name)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm + ("-batch" if batch_mode else ""))
        data = definition.arguments
        d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".csv")
    return os.path.join(*d)


def store_insert_update_delete_latencies(
    dataset_name: str,
    count: int,
    definition: Definition,
    insert_latencies: list,
    update_latencies: list,
    delete_latencies: list
):
    """
    Stores insert, update, and delete latencies for an algorithm (and hyperparameters) running against a dataset in a csv file.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition): The definition of the algorithm.
        insert_latencies (list): Insert latencies to be stored.
        update_latencies (list): Update latencies to be stored.
        delete_latencies (list): Delete latencies to be stored.
    """
    filename = build_latencies_filepath(dataset_name, count, definition)
    directory, _ = os.path.split(filename)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("id,insert_latency,update_latency,delete_latency\n")
        assert len(insert_latencies) == len(update_latencies) == len(delete_latencies)
        for i, (insert_latency, update_latency, delete_latency) in enumerate(zip(insert_latencies, update_latencies, delete_latencies)):
            f.write(f"{i},{insert_latency},{update_latency},{delete_latency}\n")
        avg_insert_latency = sum(insert_latencies) / len(insert_latencies)
        avg_update_latency = sum(update_latencies) / len(update_latencies)
        avg_delete_latency = sum(delete_latencies) / len(delete_latencies)
        f.write(f"average,{avg_insert_latency},{avg_update_latency},{avg_delete_latency}\n")


def load_all_results(dataset: Optional[str] = None, 
                 count: Optional[int] = None, 
                 batch_mode: bool = False) -> Iterator[Tuple[dict, h5py.File]]:
    """
    Loads all the results from the HDF5 files in the specified path.

    Args:
        dataset (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Yields:
        tuple: A tuple containing properties as a dictionary and an h5py file object.
    """
    for root, _, files in os.walk(build_result_filepath(dataset, count)):
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            try:
                with h5py.File(os.path.join(root, filename), "r+") as f:
                    properties = dict(f.attrs)
                    if batch_mode != properties["batch_mode"]:
                        continue
                    yield properties, f
            except OSError:
                print(f"Was unable to read {filename}")
                traceback.print_exc()


def get_unique_algorithms() -> Set[str]:
    """
    Retrieves unique algorithm names from the results.

    Returns:
        set: A set of unique algorithm names.
    """
    algorithms = set()
    for batch_mode in [False, True]:
        for properties, _ in load_all_results(batch_mode=batch_mode):
            algorithms.add(properties["algo"])
    return algorithms
