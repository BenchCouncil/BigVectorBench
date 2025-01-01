""" Dataset generation and loading functions. """

import os
import random
import tarfile
from urllib.request import urlopen, urlretrieve, build_opener, install_opener
from typing import Any, Callable, Dict, Tuple
import gzip
import zipfile
import struct
import h5py
import numpy as np
from datasets import load_dataset
from sklearn import random_projection
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import lil_matrix
from implicit.datasets.lastfm import get_lastfm
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from implicit.utils import augment_inner_product_matrix

from bigvectorbench.algorithms.bruteforce.module import BruteForceBLAS


def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.

    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        headers = (
            "User-Agent",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        )
        opener = build_opener()
        opener.addheaders = [headers]
        install_opener(opener)
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str) -> Tuple[h5py.File, int]:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading,
    and the file handle and the dimension of the dataset are returned.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    if dataset_name in ANN_DATASETS or dataset_name in RANDOM_DATASETS:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
    elif dataset_name in BVB_DATASETS:
        dataset_url = f"https://huggingface.co/datasets/Patrickcode/BigVectorBench/blob/main/{dataset_name}.hdf5"
        # dataset_url = f"https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/{dataset_name}.hdf5"
    elif dataset_name in ART_DATASETS:
        dataset_url = f"https://huggingface.co/datasets/Patrickcode/BigVectorBench/blob/main/{dataset_name}.hdf5"
        # dataset_url = f"https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/blob/main/{dataset_name}.hdf5"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name},datasets should be in {DATASETS.keys()} or be created by create_datasets.py then added in ART_DATASETS.key()")
    
    try:
        download(dataset_url, hdf5_filename)
    except Exception:
        print(f"Cannot download {dataset_url}")
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name](hdf5_filename)

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    if "dimension" in hdf5_file.attrs:
        dimension = int(hdf5_file.attrs["dimension"])
    else:
        try:
            dataset_type = hdf5_file.attrs.get("type", None)
            if dataset_type == "mv-ann":
                dimension = len(hdf5_file["train"][0][0])
            elif dataset_type == "filter-ann":
                dimension = len(hdf5_file["train_vec"][0])
            else:
                # "sparse", "dense", "ann" and "mm-ann"
                dimension = len(hdf5_file["train"][0])
        except Exception as exc:
            raise ValueError("Could not determine dimension of dataset") from exc
    return hdf5_file, dimension


def write_output(
    train: np.ndarray,
    test: np.ndarray,
    fn: str,
    distance: str,
    point_type: str = "float",
    count: int = 100,
) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes
    and stores the nearest neighbors and their distances for the test set using a
    brute-force approach.

    Args:
        train (np.ndarray): The training data.
        test (np.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for
            each point in the test set. Defaults to 100.
    """
    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train[0])
        f.attrs["point_type"] = point_type
        print(f"train size: {train.shape[0]} * {train.shape[1]}")
        print(f"test size:  {test.shape[0]} * {test.shape[1]}")
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=train.dtype)
        bf.fit(train)

        for i, x in enumerate(test):
            if i % 1000 == 0:
                print(f"{i}/{len(test)}...")

            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


def write_filter_output(
    fn: str,
    train_vec: np.ndarray,
    test_vec: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    distance: str,
    filter_expr_func: str,
    label_names: list[str],
    label_types: list[str],
    point_type: str = "float",
    count: int = 100,
) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes
    and stores the nearest neighbors and their distances for the test set using a
    brute-force approach.

    Args:
        fn (str): The name of the HDF5 file to which data should be written.
        train_vec (np.ndarray): The training data.
        test_vec (np.ndarray): The testing data.
        train_label (np.ndarray): The training labels.
        test_label (np.ndarray): The testing labels.
        distance (str): The distance metric to use for computing nearest neighbors.
        filter_expr_func (str): The filter expression function.
        label_names (list[str]): The names of the labels.
        label_types (list[str]): The types of the labels.
        point_type (str, optional): The type of the data points. Defaults to "float".
        count (int, optional): The number of nearest neighbors to compute for
            each point in the test set. Defaults to 100.
    """
    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "filter-ann"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train_vec[0])
        f.attrs["point_type"] = point_type
        f.attrs["label_names"] = label_names
        f.attrs["label_types"] = label_types
        f.attrs["filter_expr_func"] = filter_expr_func
        print(f"train size: {train_vec.shape[0]} * {train_vec.shape[1]}")
        print(f"test size:  {test_vec.shape[0]} * {test_vec.shape[1]}")
        f.create_dataset("train_vec", data=train_vec)
        f.create_dataset("test_vec", data=test_vec)
        f.create_dataset("train_label", data=train_label)
        f.create_dataset("test_label", data=test_label)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test_vec), count), dtype=int)
        distances_ds = f.create_dataset(
            "distances", (len(test_vec), count), dtype=float
        )

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=train_vec.dtype)
        bf.fit(train_vec, train_label, label_names, label_types)

        # Execute the filter expression function
        exec(filter_expr_func, globals())
        filter_expr = globals()["filter_expr"]

        # from concurrent.futures import ThreadPoolExecutor, as_completed

        # def process_query(i, x, labels, count, filter_expr, bf):
        #     expr = filter_expr(*labels)
        #     res = list(bf.query_with_distances(x, count, expr))
        #     res.sort(key=lambda t: t[-1])
        #     return i, [idx for idx, _ in res], [dist for _, dist in res]

        # with ThreadPoolExecutor(max_workers=1000) as executor:
        #     future_to_index = {
        #         executor.submit(process_query, i, x, labels, count, filter_expr, bf): i
        #         for i, (x, labels) in enumerate(zip(test_vec, test_label))
        #     }

        #     for future in as_completed(future_to_index):
        #         i, neighbors, distances = future.result()
        #         neighbors_ds[i] = neighbors
        #         distances_ds[i] = distances

        #         if i % 1000 == 0:
        #             print(f"{i}/{len(test_vec)}...")

        for i, (x, labels) in enumerate(zip(test_vec, test_label)):
            if i % 1000 == 0:
                print(f"{i}/{len(test_vec)}...")

            # Query the model and sort results by distance
            expr = filter_expr(*labels)
            res = list(bf.query_with_distances(x, count, expr))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


def write_sparse_output(
    train: np.ndarray,
    test: np.ndarray,
    fn: str,
    distance: str,
    dimension: int,
    count: int = 100,
) -> None:
    """
    Writes the provided sparse training and testing data to an HDF5 file. It also computes
    and stores the nearest neighbors and their distances for the test set using a
    brute-force approach.

    Args:
        train (np.ndarray): The sparse training data.
        test (np.ndarray): The sparse testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        dimension (int): The dimensionality of the data.
        neighbors_count (int, optional): The number of nearest neighbors to compute for
            each point in the test set. Defaults to 100.
    """
    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "sparse"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = dimension
        f.attrs["point_type"] = "bit"
        print(f"train size: {train.shape[0]} * {dimension}")
        print(f"test size:  {test.shape[0]} * {dimension}")

        # Ensure the sets are sorted
        train = np.array([sorted(t) for t in train])
        test = np.array([sorted(t) for t in test])

        # Flatten and write train and test sets
        flat_train = np.concatenate(train)
        flat_test = np.concatenate(test)
        f.create_dataset("train", data=flat_train)
        f.create_dataset("test", data=flat_test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Write sizes of train and test sets
        f.create_dataset("size_train", data=[len(t) for t in train])
        f.create_dataset("size_test", data=[len(t) for t in test])

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=flat_train.dtype)
        bf.fit(train)

        for i, x in enumerate(test):
            if i % 1000 == 0:
                print(f"{i}/{len(test)}...")
            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


def random_float(
    out_fn: str, n_dims: int, n_samples: int, centers: int, distance: str
) -> None:
    """random-float"""
    X = make_blobs(
        n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1
    )[0]
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def random_bitstring(out_fn: str, n_dims: int, n_samples: int, n_queries: int) -> None:
    """random-bitstring"""
    Y = make_blobs(
        n_samples=n_samples, n_features=n_dims, centers=n_queries, random_state=1
    )[0]
    X = np.zeros((n_samples, n_dims), dtype=np.bool_)
    for i, vec in enumerate(Y):
        X[i] = np.array([v > 0 for v in vec], dtype=np.bool_)
    X_train, X_test = train_test_split(X, test_size=n_queries)
    write_output(X_train, X_test, out_fn, "hamming", "bit")


def random_jaccard(
    out_fn: str, n: int = 10000, size: int = 50, universe: int = 80
) -> None:
    """random jaccard dataset"""
    random.seed(1)
    l = list(range(universe))
    X = []
    for _ in range(n):
        X.append(random.sample(l, size))

    X_train, X_test = train_test_split(np.array(X), test_size=100)
    write_sparse_output(X_train, X_test, out_fn, "jaccard", universe)


def random_filter(
    out_fn: str,
    n_dims: int,
    n_samples: int,
    centers: int,
    n_filters: int,
    distance: str = "euclidean",
) -> None:
    """Gen random filter dataset with n_filters filters"""
    X = make_blobs(
        n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1
    )[0]
    label_names = [f"label_{i}" for i in range(n_filters)]
    label_types = ["int32" for i in range(n_filters)]
    print(f"labels_names: {label_names}")
    print(f"labels_types: {label_types}")
    filter_expr = " and ".join(
        [
            f"{label_name} <= " + "{" + f"{label_name}" + "}"
            for label_name in label_names
        ]
    )
    print(f"filter_expr: {filter_expr}")
    filter_expr_func = "def filter_expr(" + ", ".join(label_names) + "):\n"
    filter_expr_func += '    return f"' + filter_expr + '"\n'
    print(f"filter_expr_func: {filter_expr_func}")
    filters = np.random.randint(0, 100, (n_samples, n_filters))
    X_train, X_test, train_label, test_label = train_test_split(
        X, filters, test_size=0.1, random_state=42
    )
    write_filter_output(
        out_fn,
        X_train,
        X_test,
        train_label,
        test_label,
        distance,
        filter_expr_func,
        label_names,
        label_types,
    )


def random_mv(out_fn: str) -> None:
    """random multi-vector dataset"""
    print(f"preparing {out_fn}")
    n = 10000
    d = 100
    X = np.random.rand(n, 4, d)
    print(f"data size: {X.shape[0]} * {X.shape[1]} * {X.shape[2]}")
    X_train, X_test = train_test_split(X, test_size=1000, random_state=42)
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)
    print(f"train size: {X_train.shape[0]} * {X_train.shape[1]} * {X_train.shape[2]}")
    print(f"test size: {X_test.shape[0]} * {X_test.shape[1]} * {X_test.shape[2]}")
    # (1000, 1, 4, 100) - (1, 9000, 4, 100) => (1000, 9000, 4, 100)
    distance_matrix = np.mean(
        np.linalg.norm(X_test[:, np.newaxis] - X_train, axis=3), axis=2
    )
    print(
        f"distance matrix size: {distance_matrix.shape[0]} * {distance_matrix.shape[1]}"
    )
    nearest_indices = np.argpartition(distance_matrix, 100, axis=1)[:, :100]
    print(
        f"nearest indices size: {nearest_indices.shape[0]} * {nearest_indices.shape[1]}"
    )
    nearest_indices = nearest_indices[
        np.arange(nearest_indices.shape[0])[:, None],
        np.argsort(
            distance_matrix[
                np.arange(distance_matrix.shape[0])[:, None], nearest_indices
            ],
            axis=1,
        ),
    ]
    nearest_distances = np.sort(distance_matrix, axis=1)[:, :100]
    with h5py.File(out_fn, "w") as f:
        f.attrs["type"] = "mv-ann"
        f.attrs["distance"] = "euclidean"
        f.create_dataset("train", data=X_train, dtype="float32")
        f.create_dataset("test", data=X_test, dtype="float32")
        f.create_dataset("neighbors", data=nearest_indices, dtype="int32")
        f.create_dataset("distances", data=nearest_distances, dtype="float32")


def glove(out_fn: str, d: int) -> None:
    """glove"""
    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print(f"preparing {out_fn}")
        z_fn = f"glove.twitter.27B.{d}.txt"
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(np.array(v))
        X_train, X_test = train_test_split(X, test_size=10000)
        write_output(np.array(X_train), np.array(X_test), out_fn, "angular")


def _load_texmex_vectors(f: Any, n: int, k: int) -> np.ndarray:
    v = np.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))
    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> np.ndarray:
    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn: str) -> None:
    """sift"""
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", "sift.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def gist(out_fn: str) -> None:
    """gist"""
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def _load_mnist_vectors(fn: str) -> np.ndarray:
    print(f"parsing vectors in {fn}...")
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = np.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append(
            [struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)]
        )
    return np.array(vectors)


def mnist(out_fn: str) -> None:
    """mnist"""
    download(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "mnist-train.gz"
    )
    download(
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "mnist-test.gz"
    )
    train = _load_mnist_vectors("mnist-train.gz")
    test = _load_mnist_vectors("mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn: str) -> None:
    """fashion-mnist"""
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "fashion-mnist-train.gz",
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "fashion-mnist-test.gz",
    )
    train = _load_mnist_vectors("fashion-mnist-train.gz")
    test = _load_mnist_vectors("fashion-mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn: str) -> None:
    """deep-image"""
    yadisk_key = "https://yadi.sk/d/11eDCm7Dsn9GA"
    response = urlopen(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        + yadisk_key
        + "&path=/deep10M.fvecs"
    )
    response_body = response.read().decode("utf-8")
    dataset_url = response_body.split(",")[0][9:-1]
    filename = os.path.join("data", "deep-image.fvecs")
    download(dataset_url, filename)
    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]
    X_train, X_test = train_test_split(fv, test_size=10000)
    write_output(X_train, X_test, out_fn, "angular")


def transform_bag_of_words(filename: str, n_dimensions: int, out_fn: str) -> None:
    """transform_bag_of_words"""
    with gzip.open(filename, "rb") as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(
            n_components=n_dimensions
        ).fit_transform(B)
        X_train, X_test = train_test_split(C, test_size=10000)
        write_output(np.array(X_train), np.array(X_test), out_fn, "angular")


def nytimes(out_fn: str, n_dimensions: int) -> None:
    """nytimes"""
    fn = f"nytimes_{n_dimensions}.txt.gz"
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz",
        fn,
    )
    transform_bag_of_words(fn, n_dimensions, out_fn)


def sift_hamming(out_fn: str, fn: str) -> None:
    """sift-hamming"""
    local_fn = fn + ".tar.gz"
    # url = f"http://web.stanford.edu/~maxlam/word_vectors/compressed/{path}/{fn}.tar.gz"
    url = f"http://sss.projects.itu.dk/bigvectorbench/datasets/{fn}.tar.gz"
    download(url, local_fn)
    print(f"parsing vectors in {local_fn}...")
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        n_words, k = [int(z) for z in next(f).strip().split()]
        X = np.zeros((n_words, k), dtype=np.bool_)
        for i in range(n_words):
            X[i] = np.array(
                [float(z) > 0 for z in next(f).strip().split()[1:]], dtype=np.bool_
            )

        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def kosarak(out_fn: str) -> None:
    """kosarak"""
    local_fn = "kosarak.dat.gz"
    # only consider sets with at least min_elements many elements
    min_elements = 20
    url = f"http://fimi.uantwerpen.be/data/{local_fn}"
    download(url, local_fn)

    X = []
    dimension = 0
    with gzip.open("kosarak.dat.gz", "r") as f:
        content = f.readlines()
        # preprocess data to find sets with more than 20 elements
        # keep track of used ids for reenumeration
        for line in content:
            if len(line.split()) >= min_elements:
                X.append(list(map(int, line.split())))
                dimension = max(dimension, max(X[-1]) + 1)

    X_train, X_test = train_test_split(np.array(X), test_size=500)
    write_sparse_output(X_train, X_test, out_fn, "jaccard", dimension)


def lastfm(out_fn: str, n_dimensions: int, test_size: int = 50000) -> None:
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = AlternatingLeastSquares(factors=n_dimensions)
    model.fit(bm25_weight(play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = np.append(
        model.user_factors, np.zeros((model.user_factors.shape[0], 1)), axis=1
    )

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, user_factors, out_fn, "angular")


def movielens(
    fn: str,
    ratings_file: str,
    out_fn: str,
    separator: str = "::",
    ignore_header: bool = False,
) -> None:
    """movielens"""
    url = f"http://files.grouplens.org/datasets/movielens/{fn}"
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        file = z.open(ratings_file)
        if ignore_header:
            file.readline()

        print(f"preparing {out_fn}")

        users = {}
        X = []
        dimension = 0
        for line in file:
            el = line.decode("UTF-8").split(separator)

            userId = el[0]
            itemId = int(el[1])
            rating = float(el[2])

            if rating < 3:  # We only keep ratings >= 3
                continue

            if userId not in users:
                users[userId] = len(users)
                X.append([])

            X[users[userId]].append(itemId)
            dimension = max(dimension, itemId + 1)

        X_train, X_test = train_test_split(np.array(X), test_size=500)
        write_sparse_output(X_train, X_test, out_fn, "jaccard", dimension)


def movielens1m(out_fn: str) -> None:
    """movielens-1m"""
    movielens("ml-1m.zip", "ml-1m/ratings.dat", out_fn)


def movielens10m(out_fn: str) -> None:
    """movielens-10m"""
    movielens("ml-10m.zip", "ml-10M100K/ratings.dat", out_fn)


def movielens20m(out_fn: str) -> None:
    """movielens-20m"""
    movielens("ml-20m.zip", "ml-20m/ratings.csv", out_fn, ",", True)


def dbpedia_entities_openai_ada002_1M(out_fn, n=None):
    """dbpedia-entities-openai-ada002-1M"""
    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")
    if n is not None and n >= 100_000:
        data = data.select(range(n))

    embeddings = data.to_pandas()["openai"].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    X_train, X_test = train_test_split(embeddings, test_size=10_000, random_state=42)

    write_output(X_train, X_test, out_fn, "angular")


def bvb_dataset(out_fn: str, dataset_name: str) -> None:
    """
    bvb_dataset: Downloads a dataset from the BigVectorBench repository on Hugging Face Datasets Hub
    """
    dataset_url = f"https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/{dataset_name}.hdf5"
    # dataset_url = f"https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/{dataset_name}.hdf5"
    download(dataset_url, out_fn)


def dbpedia_entities_openai3_text_embedding_3_large_3072_1M(out_fn, i, distance):
    """dbpedia-entities-openai3-text-embedding-3-large-3072-1M"""
    data = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M", split="train"
    )
    if i is not None and i >= 100_000:
        data = data.select(range(i))
    embeddings = data.to_pandas()["text-embedding-3-large-3072-embedding"].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 3072))
    X_train, X_test = train_test_split(embeddings, test_size=10_000, random_state=42)
    write_output(X_train, X_test, out_fn, distance)


def dbpedia_entities_openai3_text_embedding_3_large_1536_1M(out_fn, i, distance):
    """dbpedia-entities-openai3-text-embedding-3-large-1536-1M"""
    data = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M", split="train"
    )
    if i is not None and i >= 100_000:
        data = data.select(range(i))
    embeddings = data.to_pandas()["text-embedding-3-large-1536-embedding"].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))
    X_train, X_test = train_test_split(embeddings, test_size=10_000, random_state=42)
    write_output(X_train, X_test, out_fn, distance)


RANDOM_DATASETS: Dict[str, Callable[[str], None]] = {
    "random-xs-20-euclidean": lambda out_fn: random_float(
        out_fn, 20, 10000, 100, "euclidean"
    ),
    "random-xs-32-euclidean": lambda out_fn: random_float(
        out_fn, 32, 10000, 100, "euclidean"
    ),
    "random-s-100-euclidean": lambda out_fn: random_float(
        out_fn, 100, 100000, 1000, "euclidean"
    ),
    "random-xs-20-angular": lambda out_fn: random_float(
        out_fn, 20, 10000, 100, "angular"
    ),
    "random-xs-32-angular": lambda out_fn: random_float(
        out_fn, 32, 10000, 100, "angular"
    ),
    "random-s-100-angular": lambda out_fn: random_float(
        out_fn, 100, 100000, 1000, "angular"
    ),
    "random-xs-16-hamming": lambda out_fn: random_bitstring(out_fn, 16, 10000, 100),
    "random-s-128-hamming": lambda out_fn: random_bitstring(out_fn, 128, 50000, 1000),
    "random-l-256-hamming": lambda out_fn: random_bitstring(out_fn, 256, 100000, 1000),
    "random-s-jaccard": lambda out_fn: random_jaccard(
        out_fn, n=10000, size=20, universe=40
    ),
    "random-l-jaccard": lambda out_fn: random_jaccard(
        out_fn, n=100000, size=70, universe=100
    ),
    "random-mv": random_mv,
    "random-xs-32-euclidean-2filter": lambda out_fn: random_filter(
        out_fn, 32, 10000, 100, 2, "euclidean"
    ),
}


ANN_DATASETS: Dict[str, Callable[[str], None]] = {
    "deep-image-96-angular": deep_image,
    "fashion-mnist-784-euclidean": fashion_mnist,
    "gist-960-euclidean": gist,
    "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "mnist-784-euclidean": mnist,
    "sift-128-euclidean": sift,
    "nytimes-256-angular": lambda out_fn: nytimes(out_fn, 256),
    "nytimes-16-angular": lambda out_fn: nytimes(out_fn, 16),
    "lastfm-64-dot": lambda out_fn: lastfm(out_fn, 64),
    "sift-256-hamming": lambda out_fn: sift_hamming(out_fn, "sift.hamming.256"),
    "kosarak-jaccard": kosarak,
    "movielens1m-jaccard": movielens1m,
    "movielens10m-jaccard": movielens10m,
    "movielens20m-jaccard": movielens20m,
}
ANN_DATASETS.update(
    {
        f"dbpedia-openai-ada002-{n//1000}k-angular": lambda out_fn, i=n: dbpedia_entities_openai_ada002_1M(
            out_fn, i
        )
        for n in range(100_000, 1_100_000, 100_000)
    }
)


BVB_DATASETS: Dict[str, Callable[[str], None]] = {
    "ag_news-384-euclidean": lambda out_fn: bvb_dataset(
        out_fn, "ag_news-384-euclidean"
    ),
    "ag_news-384-euclidean-filter": lambda out_fn: bvb_dataset(
        out_fn, "ag_news-384-euclidean-filter"
    ),
    "cc_news-384-euclidean": lambda out_fn: bvb_dataset(
        out_fn, "cc_news-384-euclidean"
    ),
    "cc_news-384-euclidean-filter": lambda out_fn: bvb_dataset(
        out_fn, "cc_news-384-euclidean-filter"
    ),
    "app_reviews-384-euclidean": lambda out_fn: bvb_dataset(
        out_fn, "app_reviews-384-euclidean"
    ),
    "app_reviews-384-euclidean-filter": lambda out_fn: bvb_dataset(
        out_fn, "app_reviews-384-euclidean-filter"
    ),
    "amazon-384-euclidean": lambda out_fn: bvb_dataset(out_fn, "amazon-384-euclidean"),
    "amazon-384-euclidean-1filter": lambda out_fn: bvb_dataset(
        out_fn, "amazon-384-euclidean-1filter"
    ),
    "amazon-384-euclidean-5filter": lambda out_fn: bvb_dataset(
        out_fn, "amazon-384-euclidean-5filter"
    ),
    "gpt4vision-1024-euclidean-mm": lambda out_fn: bvb_dataset(
        out_fn, "gpt4vision-1024-euclidean-mm"
    ),
    "librispeech_asr-1024-euclidean-mm": lambda out_fn: bvb_dataset(
        out_fn, "librispeech_asr-1024-euclidean-mm"
    ),
    "librispeech_asr-1024-euclidean-mm-asr": lambda out_fn: bvb_dataset(
        out_fn, "librispeech_asr-1024-euclidean-mm-asr"
    ),
    "img-wikipedia-1024-euclidean-mm": lambda out_fn: bvb_dataset(
        out_fn, "img-wikipedia-1024-euclidean-mm"
    ),
    "img-wikipedia-1024-euclidean-mm-ocr": lambda out_fn: bvb_dataset(
        out_fn, "img-wikipedia-1024-euclidean-mm-ocr"
    ),
    "webvid-4-512-euclidean": lambda out_fn: bvb_dataset(
        out_fn, "webvid-4-512-euclidean"
    ),
}

BVB_DATASETS.update(
    {
        f"dbpedia-entities-openai3-text-embedding-3-large-3072-{n//1000}k-{distance}": lambda out_fn, i=n, d=distance: dbpedia_entities_openai3_text_embedding_3_large_3072_1M(
            out_fn, i, d
        )
        for n in range(100_000, 1_100_000, 100_000)
        for distance in ["angular", "euclidean"]
    },
)
BVB_DATASETS.update(
    {
        f"dbpedia-entities-openai3-text-embedding-3-large-1536-{n//1000}k-{distance}": lambda out_fn, i=n, d=distance: dbpedia_entities_openai3_text_embedding_3_large_1536_1M(
            out_fn, i, d
        )
        for n in range(100_000, 1_100_000, 100_000)
        for distance in ["angular", "euclidean"]
    }
)

def artificial_dataset(out_fn: str, dataset_name: str) -> None:
    """
    bvb_dataset: Downloads a dataset from the BigVectorBench repository on Hugging Face Datasets Hub
    """
    dataset_url = f"https://huggingface.co/datasets/AnnaZh/Bigvectorbench-artificial-datasets/resolve/main/{dataset_name}.hdf5"
    # dataset_url = f"https://hf-mirror.com/datasets/AnnaZh/Bigvectorbench-artificial-datasets/resolve/main/{dataset_name}.hdf5"
    download(dataset_url, out_fn)

ART_DATASETS: Dict[str, Callable[[str], None]] = {
    "deep1M-2filter-50a": lambda out_fn: artificial_dataset(
        out_fn, "deep1M-2filter-50a"
    ),
    "msong-1filter-80a": lambda out_fn: artificial_dataset(
        out_fn, "msong-1filter-80a"
    ),
    "sift10m-6filter-6a": lambda out_fn: artificial_dataset(
        out_fn, "sift10m-6filter-6a"
    ),
    "tiny5m-6filter-12a": lambda out_fn: artificial_dataset(
        out_fn, "tiny5m-6filter-12a"
    )
}

DATASETS: Dict[str, Callable[[str], None]] = {}
DATASETS.update(RANDOM_DATASETS)
DATASETS.update(ANN_DATASETS)
DATASETS.update(BVB_DATASETS)
DATASETS.update(ART_DATASETS)