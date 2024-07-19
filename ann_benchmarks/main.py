""" Main module for running the benchmarking process. """
import argparse
from dataclasses import replace
import logging
import logging.config
import multiprocessing.pool
import os
import random
import shutil
import sys
from typing import List

import docker
import psutil

from .definitions import (Definition, InstantiationStatus, algorithm_status,
                                     get_definitions, list_algorithms)
from .constants import INDEX_DIR
from .datasets import DATASETS, get_dataset
from .results import build_result_filepath
from .runner import run, run_docker


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("bvb")


def positive_int(input_str: str) -> int:
    """
    Validates if the input string can be converted to a positive integer.

    Args:
        input_str (str): The input string to validate and convert to a positive integer.

    Returns:
        int: The validated positive integer.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to a positive integer.
    """
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer") from exc

    return i


def memory_type(value):
    """
    Validates if the input string can be converted to a memory size.

    Args:
        value (str): The input string to validate and convert to a memory size.

    Returns:
        int or str: The validated memory size.
    """
    try:
        return int(value)
    except ValueError:
        return value


def parse_mem_limit(mem_limit):
    """
    Parses the memory limit string and returns the memory limit in bytes.

    Args:
        mem_limit (str | int | float): The string containing the memory limit.

    Returns:
        int: The memory limit in bytes.
    """
    units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}

    if isinstance(mem_limit, (int, float)):
        return int(mem_limit)

    if isinstance(mem_limit, str):
        mem_limit = mem_limit.strip().lower()
        if mem_limit[-1] in units:
            return int(float(mem_limit[:-1]) * units[mem_limit[-1]])
        else:
            return int(float(mem_limit))  # Assume bytes if no unit specified

    raise ValueError(f"Invalid memory limit format: {mem_limit}")


def parse_cpu_set(cpu_set_string):
    """
    Parses the CPU set string and returns a sorted set of CPU numbers.

    Args:
        cpu_set_string (str): The string containing the CPU numbers.

    Returns:
        set: A sorted set of CPU numbers.
    """
    cpu_set = set()
    parts = cpu_set_string.split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            cpu_set.update(range(int(start), int(end) + 1))
        else:
            cpu_set.add(int(part))
    return sorted(cpu_set)


def run_worker(cpuset_cpus : str, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    """
    Executes the algorithm based on the provided parameters.

    The algorithm is either executed directly or through a Docker container based on the `args.local`
     argument. The function runs until the queue is emptied. When running in a docker container, it 
    executes the algorithm in a Docker container.

    Args:
        # cpu (int): The CPU number to be used in the execution.
        cpuset_cpus (str): The CPUs in which to allow the docker container to run.
        args (argparse.Namespace): User provided arguments for running workers. 
        queue (multiprocessing.Queue): The multiprocessing queue that contains the algorithm definitions.

    Returns:
        None
    """
    while not queue.empty():
        definition = queue.get()
        if args.local:
            run(definition, args.dataset, args.count, args.runs, args.batch)
        else:
            run_docker(
                definition, args.dataset, args.count, args.runs, args.timeout, args.batch, cpuset_cpus, args.memory
            )


def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset to load training points from",
        default="glove-100-angular",
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "-k",
        "--count",
        default=10,
        type=positive_int,
        help="the number of near neighbours to search for"
    )
    parser.add_argument(
        "--definitions",
        metavar="FOLDER",
        help="base directory of algorithms. Algorithm definitions expected at 'FOLDER/*/config.yml'",
        default="ann_benchmarks/algorithms"
    )
    parser.add_argument(
        "--algorithm",
        metavar="NAME",
        help="run only the named algorithm",
        default=None
    )
    parser.add_argument(
        "--docker-tag",
        metavar="NAME",
        help="run only algorithms in a particular docker image",
        default=None
    )
    parser.add_argument(
        "--list-algorithms",
        help="print the names of all known algorithms and exit",
        action="store_true"
    )
    parser.add_argument(
        "--force",
        help="re-run algorithms even if their results already exist",
        action="store_true"
    )
    parser.add_argument(
        "--runs",
        metavar="COUNT",
        type=positive_int,
        help="run each algorithm instance %(metavar)s times and use only the best result",
        default=5,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout (in seconds) for each individual algorithm run, or -1 if no timeout should be set",
        default=10 * 3600,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, then will run everything locally (inside the same process) rather than using Docker",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If set, algorithms get all queries at once"
    )
    parser.add_argument(
        "--max-n-algorithms",
        type=int,
        help="Max number of algorithms to run (just used for testing)",
        default=-1
    )
    parser.add_argument(
        "--run-disabled",
        help="run algorithms that are disabled in algos.yml",
        action="store_true"
    )
    parser.add_argument(
        "--parallelism",
        type=positive_int,
        help="Number of Docker containers in parallel",
        default=1
    )
    parser.add_argument(
        "--cpuset-cpus",
        metavar="CPUSET",
        help="The CPUs in which to allow the container to run(e.g., 0-2, 0,1) (only used in Docker mode), default is all CPUs",
        default=None
    )
    parser.add_argument(
        "--memory",
        type=memory_type,
        help="Memory limit for Docker containers, default is all available memory",
        default=None
    )
    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None
    return args


def filter_already_run_definitions(
    definitions: List[Definition],
    dataset: str,
    count: int,
    batch: bool,
    force: bool
) -> List[Definition]:
    """Filters out the algorithm definitions based on whether they have already been run or not.

    This function checks if there are existing results for each definition by constructing the 
    result filename from the algorithm definition and the provided arguments. If there are no 
    existing results or if the parameter `force=True`, the definition is kept. Otherwise, it is
    discarded.

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.
        dataset (str): The name of the dataset to load training points from.
        force (bool): If set, re-run algorithms even if their results already exist.

        count (int): The number of near neighbours to search for (only used in file naming convention).
        batch (bool): If set, algorithms get all queries at once (only used in file naming convention).

    Returns:
        List[Definition]: A list of algorithm definitions that either have not been run or are 
                          forced to be re-run.
    """
    filtered_definitions = []

    for definition in definitions:
        not_yet_run = [
            query_args
            for query_args in (definition.query_argument_groups or [[]])
            if force or not os.path.exists(build_result_filepath(dataset, count, definition, query_args, batch))
        ]

        if not_yet_run:
            definition = replace(definition, query_argument_groups=not_yet_run) if definition.query_argument_groups else definition
            filtered_definitions.append(definition)

    return filtered_definitions


def filter_by_available_docker_images(definitions: List[Definition]) -> List[Definition]:
    """
    Filters out the algorithm definitions that do not have an associated, available Docker images.

    This function uses the Docker API to list all Docker images available in the system. It 
    then checks the Docker tags associated with each algorithm definition against the list 
    of available Docker images, filtering out those that are unavailable. 

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.

    Returns:
        List[Definition]: A list of algorithm definitions that are associated with available Docker images.
    """
    docker_client = docker.from_env()
    docker_tags = {tag.split(":")[0] for image in docker_client.images.list() for tag in image.tags}

    missing_docker_images = set(d.docker_tag for d in definitions).difference(docker_tags)
    if missing_docker_images:
        logger.info("not all docker images available, only: %s", docker_tags)
        logger.info("missing docker images: %s", missing_docker_images)
        definitions = [d for d in definitions if d.docker_tag in docker_tags]

    return definitions


def check_module_import_and_constructor(df: Definition) -> bool:
    """
    Verifies if the algorithm module can be imported and its constructor exists.

    This function checks if the module specified in the definition can be imported. 
    Additionally, it verifies if the constructor for the algorithm exists within the 
    imported module.

    Args:
        df (Definition): A definition object containing the module and constructor 
        for the algorithm.

    Returns:
        bool: True if the module can be imported and the constructor exists, False 
        otherwise.
    """
    status = algorithm_status(df)
    if status == InstantiationStatus.NO_CONSTRUCTOR:
        raise ImportError(
            f"{df.module}.{df.constructor}({df.arguments}): error: the module '{df.module}' does not expose the named constructor"
        )
    if status == InstantiationStatus.NO_MODULE:
        logging.warning(
            "%s.%s(%s): the module '%s' could not be loaded; skipping", df.module, df.constructor, df.arguments, df.module
        )
        return False

    return True

def create_workers_and_execute(definitions: List[Definition], args: argparse.Namespace):
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        args (argparse.Namespace): User provided arguments for running workers. 

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than 
                   one worker.
    """
    total_cpu_num = multiprocessing.cpu_count()
    cpu_set = parse_cpu_set(args.cpuset_cpus) if args.cpuset_cpus else list(range(total_cpu_num))
    cpu_count = len(cpu_set)
    if cpu_set[-1] >= total_cpu_num:
        raise ValueError(f"CPU number {cpu_set[-1]} is larger than the number of CPUs available ({total_cpu_num})")

    if args.parallelism > cpu_count:
        raise ValueError(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    if args.batch and args.parallelism > 1:
        raise ValueError(
            f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {args.parallelism})"
        )

    if cpu_count % args.parallelism != 0:
        raise ValueError(f"Number of CPUs ({cpu_count}) must be divisible by parallelism ({args.parallelism})")
    else:
        cpu_sets = [cpu_set[i * (cpu_count // args.parallelism):(i + 1) * (cpu_count // args.parallelism)] for i in range(args.parallelism)]
        cpu_sets = [",".join(map(str, cpu_set)) for cpu_set in cpu_sets]

    memory_available = psutil.virtual_memory().available
    memory_limit = parse_mem_limit(args.memory) if args.memory else memory_available
    if args.parallelism * memory_limit > memory_available:
        raise ValueError(f"Memory limit {args.memory} per container times parallelism {args.parallelism} exceeds available memory {memory_available}")

    task_queue = multiprocessing.Queue()
    for definition in definitions:
        task_queue.put(definition)

    try:
        workers = [multiprocessing.Process(target=run_worker, args=(cpu_sets[i], args, task_queue)) for i in range(args.parallelism)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        logger.info("Terminating %d workers", len(workers))
        for worker in workers:
            worker.terminate()


def filter_disabled_algorithms(definitions: List[Definition]) -> List[Definition]:
    """
    Excludes disabled algorithms from the given list of definitions.

    This function filters out the algorithm definitions that are marked as disabled in their `config.yml`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.

    Returns:
        List[Definition]: A list of algorithm definitions excluding any that are disabled.
    """
    disabled_algorithms = [d for d in definitions if d.disabled]
    if disabled_algorithms:
        logger.info("Not running disabled algorithms %s", disabled_algorithms)

    return [d for d in definitions if not d.disabled]


def limit_algorithms(definitions: List[Definition], limit: int) -> List[Definition]:
    """
    Limits the number of algorithm definitions based on the given limit.

    If the limit is negative, all definitions are returned. For valid 
    sampling, `definitions` should be shuffled before `limit_algorithms`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.
        limit (int): The maximum number of definitions to return.

    Returns:
        List[Definition]: A trimmed list of algorithm definitions.
    """
    return definitions if limit < 0 else definitions[:limit]


def main():
    """
    Main function that orchestrates the execution of the benchmarking process.
    """
    args = parse_arguments()

    if args.list_algorithms:
        list_algorithms(args.definitions)
        sys.exit(0)

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    dataset, dimension = get_dataset(args.dataset)
    definitions: List[Definition] = get_definitions(
        dimension=dimension,
        point_type=dataset.attrs.get("point_type", "float"),
        distance_metric=dataset.attrs["distance"],
        count=args.count,
        base_dir=args.definitions,
    )
    random.shuffle(definitions)

    definitions = filter_already_run_definitions(
        definitions,
        dataset=args.dataset,
        count=args.count,
        batch=args.batch,
        force=args.force,
    )

    if args.algorithm:
        logger.info("running only %s", args.algorithm)
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    if not args.local:
        definitions = filter_by_available_docker_images(definitions)
    else:
        definitions = list(filter(
            check_module_import_and_constructor, definitions
        ))

    definitions = filter_disabled_algorithms(definitions) if not args.run_disabled else definitions
    definitions = limit_algorithms(definitions, args.max_n_algorithms)

    if len(definitions) == 0:
        raise ValueError("Nothing to run")
    else:
        logger.info("Order: %s", definitions)

    create_workers_and_execute(definitions, args)
