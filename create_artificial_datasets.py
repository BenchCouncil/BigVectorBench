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
from tqdm import tqdm
import time
import math
# from datasets import load_dataset
from sklearn import random_projection
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from implicit.datasets.lastfm import get_lastfm
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from implicit.utils import augment_inner_product_matrix
import argparse

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
        raise argparse.ArgumentTypeError(
            f"{input_str} is not a positive integer"
        ) from exc

    return i

def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--type",
        metavar="TYPE",
        help="the dataset to be generated as",
        default="random",
        choices={"random","average"},
    )
    parser.add_argument(
        "--n",
        default=10000,
        type=positive_int,
        help="the number of train data to be generated",
    )
    parser.add_argument(
        "--m",
        default=1000,
        type=positive_int,
        help="the number of test data to be generated",
    )
    parser.add_argument(
        "--a",
        default=80,
        type=positive_int,
        help="a'%' is the ratio of data to be filtered",
    )
    parser.add_argument(
        "--d",
        default=128,
        type=positive_int,
        help="the dimension of data to be generated",
    )
    parser.add_argument(
        "--l",
        default=1,
        type=positive_int,
        help="the number of labels for data to be generated",
    )
    parser.add_argument(
        "--path",
        default="",
        help="the creates dataset will be at 'data/path/***.hdf5'",
    )
    parser.add_argument(
        "--center",
        default=100,
        type=positive_int,
        help="the number of centers for data to be generated",
    )
    parser.add_argument(
        "--metric",
        default="inner_product",
        help="the metric type for distance to be calculated",
    )
    parser.add_argument(
        "--maxlabel",
        default=100000,
        type=positive_int,
        help="the max label value to be generated",
    )
    parser.add_argument(
        "--topk",
        default=200,
        type=positive_int,
        help="the topk neighbors to be retriveled",
    )

    args = parser.parse_args()

    return args

def inner_product_metric(u, v):
    return -np.dot(u, v)

def metric_mapping(_metric: str):
    """
    Mapping metric type to milvus metric type

    Args:
        _metric (str): metric type

    Returns:
        str: milvus metric type
    """
    _metric = _metric.lower()
    _metric_type = {"angular": "cosine", "euclidean": "euclidean","inner_product":inner_product_metric}.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


def filter_knn_calculate(
    distance:str,
    train_vec: np.ndarray,
    test_vec: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    topk:int,
    ratio_request: float,
) -> None:
    neighbors_ds = np.full((len(test_vec), topk), -1,   dtype=np.int32)
    distances_ds = np.full((len(test_vec), topk), -1.0, dtype=np.float32)
    maxcnt = 0
    mincnt = train_vec.shape[0]
    sumcnt = 0
    if ratio_request >= 0.2:
        for i, qry in tqdm(enumerate(test_vec),desc="Processing"):
            vec_set = []
            idx_set = []
            qry_label_left = []
            qry_label_right = []

            n_samples_fit = min(int(topk*1.5/ratio_request), train_vec.shape[0])
            nn = NearestNeighbors(n_neighbors=n_samples_fit, metric=metric_mapping(distance), n_jobs=-1,algorithm='brute')
            nn.fit(train_vec)
            distances, indices = nn.kneighbors(np.array([qry]))

            neighbors_tmp = np.full((int(topk*1.5/ratio_request)), -1,   dtype=np.int32)
            distances_tmp = np.full((int(topk*1.5/ratio_request)), -1.0, dtype=np.float32)
            labels_tmp = np.full((int(topk*1.5/ratio_request),train_label.shape[1]), 0, dtype=np.int32)

            neighbors_tmp[:n_samples_fit] = indices[0]
            distances_tmp[:n_samples_fit] = distances[0]
            labels_tmp[:n_samples_fit] = train_label[indices[0]]

            flags = np.zeros(n_samples_fit, dtype=int)
            for j,j_test_label in enumerate(test_label[i]):
                # qry_label_left.append(j_test_label[0])
                # qry_label_right.append(j_test_label[1])
                left = j_test_label[0]
                right = j_test_label[1]
                flags += (left <= labels_tmp[:,j]) & (labels_tmp[:,j] <= right)

            cnt = 0
            lens = train_label.shape[1]
            for j,flag in enumerate(flags):
                if flag == lens:
                    vec_set.append(distances_tmp[j])
                    idx_set.append(neighbors_tmp[j])
                    cnt += 1

            if cnt < topk:
                print(f"Warning: {i}-th query has {cnt} neighbors")

            train_vec_scope = np.array(vec_set, dtype=np.float32)
            train_idx_scope = np.array(idx_set, dtype=np.int32)

            if i % 100 == 0:
                print(f"{i}-th query : train_vec_scope.shape: {train_vec_scope.shape}, train_idx_scope.shape: {train_idx_scope.shape}")
            
            n_samples_fit = min(topk, train_vec_scope.shape[0])
            if(n_samples_fit < 1):
                print(f"Warning: {i}-th query has {n_samples_fit} neighbors")
                continue

            if cnt > maxcnt:
                maxcnt = cnt
            if cnt < mincnt:
                mincnt = cnt
            sumcnt += cnt

            neighbors_ds[i, :topk] = train_idx_scope[:topk]
            distances_ds[i, :topk] = train_vec_scope[:topk]
        print(f"min filter ratio is {mincnt/(int(topk*2/ratio_request))}")
        print(f"max filter ratio is {maxcnt/ (int(topk*2/ratio_request)) } ")
        print(f"average filter ratio is {sumcnt/((int(topk*2/ratio_request))*test_vec.shape[0])}")
            

    else:
        for i, qry in tqdm(enumerate(test_vec),desc="Processing"):
            vec_set = []
            idx_set = []
            qry_label_left = []
            qry_label_right = []
            lens = train_label.shape[1]
            flags = np.zeros(train_label.shape[0], dtype=int)
            for j,j_test_label in enumerate(test_label[i]):
                # qry_label_left.append(j_test_label[0])
                # qry_label_right.append(j_test_label[1])
                left = j_test_label[0]
                right = j_test_label[1]
                flags += (left <= train_label[:,j]) & (train_label[:,j] <= right)

            cnt = 0

            lens = train_label.shape[1]
            for j,flag in enumerate(flags):
                if flag == lens:
                    vec_set.append(train_vec[j])
                    idx_set.append(j)
                    cnt += 1

            if cnt < topk:
                print(f"Warning: {i}-th query has {cnt} neighbors")

            train_vec_scope = np.array(vec_set, dtype=np.float32)
            train_idx_scope = np.array(idx_set, dtype=np.int32)

            if i % 100 == 0:
                print(f"{i}-th query : train_vec_scope.shape: {train_vec_scope.shape}, train_idx_scope.shape: {train_idx_scope.shape}")
            
            n_samples_fit = min(topk, train_vec_scope.shape[0])
            if(n_samples_fit < 1):
                print(f"Warning: {i}-th query has {n_samples_fit} neighbors")
                continue

            if cnt > maxcnt:
                maxcnt = cnt
            if cnt < mincnt:
                mincnt = cnt
            sumcnt += cnt

            nn = NearestNeighbors(n_neighbors=n_samples_fit, metric=metric_mapping(distance), n_jobs=-1,algorithm='brute')
            nn.fit(train_vec_scope)
            distances, indices = nn.kneighbors(np.array([qry]))

            neighbors_ds[i, :n_samples_fit] = train_idx_scope[indices[0]]
            distances_ds[i, :n_samples_fit] = distances[0]

        print(f"min filter ratio is {mincnt/(train_vec.shape[0])}")
        print(f"max filter ratio is {maxcnt/(train_vec.shape[0])}")
        print(f"average filter ratio is {sumcnt/(train_vec.shape[0]*test_vec.shape[0])}")

    if distance == "inner_product":
        distances_ds = -distances_ds
    return neighbors_ds,distances_ds

def write_groundtruth_output(
    fn: str,
    train_vec: np.ndarray,
    test_vec: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    topk: int,
    ratio_request: float,
) -> None:
    with h5py.File(fn, "w") as f:  

        neighbors,distances = filter_knn_calculate(train_vec,test_vec,train_label,test_label,topk,ratio_request)

        f.create_dataset("neighbors",data=neighbors,maxshape=(None, neighbors.shape[1]), chunks=(10000, neighbors.shape[1]), dtype=int)
        f.create_dataset("distances",data=distances,maxshape=(None, distances.shape[1]), chunks=(10000, distances.shape[1]), dtype=float)

        f.close()
    print(f"groundtruth is already:{fn}")

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
    label_ranges: list[str],
    label_range_types: list[str],
    point_type: str,
    topk: int,
    ratio_request: float,
) -> None:
    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "filter-ann"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train_vec[0])
        f.attrs["point_type"] = point_type
        f.attrs["label_names"] = label_names
        f.attrs["label_types"] = label_types
        f.attrs["label_ranges"] = label_ranges
        f.attrs["label_range_types"] = label_range_types
        f.attrs["filter_expr_func"] = filter_expr_func

        print(f"train size: {train_vec.shape[0]} * {train_vec.shape[1]}")
        print(f"test size:  {test_vec.shape[0]} * {test_vec.shape[1]}")
        f.create_dataset("train_vec", data=train_vec, maxshape=(None, train_vec.shape[1]), chunks=(10000, train_vec.shape[1]), dtype=float)
        f.create_dataset("test_vec", data=test_vec, maxshape=(None, test_vec.shape[1]), chunks=(10000, test_vec.shape[1]), dtype=float)
        
        test_label_scope = np.array(test_label, dtype=np.int32)
        print(f"train size: {train_label.shape[0]} * {train_label.shape[1]}")
        print(f"test size:  {test_label_scope.shape[0]} * {test_label_scope.shape[1]} * {test_label_scope.shape[2]}")
        f.create_dataset("train_label", data=train_label, maxshape=(None, train_label.shape[1]), chunks=(10000, train_label.shape[1]), dtype=int)
        f.create_dataset("test_label", data=test_label_scope, maxshape=(None, None, test_label_scope.shape[2]), chunks=(10000, test_label_scope.shape[1],test_label_scope.shape[2]), dtype=int)
        
        neighbors,distances = filter_knn_calculate(distance,train_vec,test_vec,train_label,test_label,topk,ratio_request)

        f.create_dataset("neighbors",data=neighbors,maxshape=(None, neighbors.shape[1]), chunks=(10000, neighbors.shape[1]), dtype=int)
        f.create_dataset("distances",data=distances,maxshape=(None, distances.shape[1]), chunks=(10000, distances.shape[1]), dtype=float)

        f.close()

    print(f"datafile is already:{fn}")


        
def generate_random_ranges(generate_type, sum, num_ranges, min_val, max_val, train_labels, ratio_request):
    label_ranges = []
    if generate_type=="average":
        init_random_ratio = ratio_request
    else: init_random_ratio = random.uniform(ratio_request * 0.95, min(1.0, ratio_request * 1.05))

    if num_ranges == 1:
        val = max_val * init_random_ratio
        min_side = min_val
        max_side = max_val - val
        
        lefts = np.random.randint(min_side,max_side,size=sum)
        for i in tqdm(range(sum),desc="Processing"):
            random_ranges = []
            left = lefts[i]
            right = left + val
            random_ranges.append((left, right))
            label_ranges.append(random_ranges)

    else:
        random_ratio = []
        random_ratio.append(init_random_ratio)

        a_ratio = []
        a_ratio.append(1.0)
        for i in range(num_ranges):
            random_ratio_i = random_ratio[i]/a_ratio[i]
            aver_random_ratio = math.pow(random_ratio_i, 1/(num_ranges-i))
            ai = random.uniform(max(aver_random_ratio * 0.7,math.pow(random_ratio_i,1/2)),min(1.0, (aver_random_ratio*1.3)))
            # print(i,random_ratio_i,aver_random_ratio,ai)
            if i==num_ranges-1: a_ratio.append(aver_random_ratio)
            else:a_ratio.append(ai)
            random_ratio.append(random_ratio_i)

        a_ratio_random = 1.0
        for i in range(num_ranges):
            a_ratio_random *= a_ratio[i+1]
        print(a_ratio)
        print(a_ratio_random)
        
        range_set = [1]
        for i in range(num_ranges):
            if range_set[i] >= max_val/1000: 
                for j in range(num_ranges-i-1):
                    range_set.append(1)
                break
            range_set.append(range_set[i]*10)
        print(range_set)

        lefts = []
        val_js = []
        for j in range(num_ranges):
            val_j = max_val / range_set[j] * min(a_ratio[j+1],1)
            min_j = min_val
            max_j = max(1, int(max_val / range_set[j] - val_j))
            lefts.append(np.random.randint(min_j,max_j,size=sum))
            val_js.append(val_j)

        for i in tqdm(range(sum),desc="Processing"):
            random_ranges = []
            for j in range(num_ranges):
                left_j = lefts[j][i]
                rigth_j = left_j + int(val_js[j])
                random_ranges.append((left_j,rigth_j))
            label_ranges.append(random_ranges)

        # for i in range(1,20):
        #     for j in range(num_ranges):
        #         print(label_ranges[i][j][0],label_ranges[i][j][1])

    return label_ranges


def create_filter(
    out_fn: str,
    generate_type: str,
    n_dims: int,
    n_samples: int,
    m_test:int,
    centers: int,
    n_filters: int,
    max_labels:int,
    ratio_request: float,
    distance: str = "inner_product",
    topk: int = 200,
) -> None:
    """Gen random filter dataset with n_filters filters"""
    print(f"now_dataset: {out_fn}")
    X = make_blobs(
        n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1
    )[0]
    train_X, test_X = train_test_split(X, test_size=m_test,random_state=42)
    # test_X = make_blobs(
    #     n_samples=m_test, n_features=n_dims, centers=centers, random_state=1
    # )[0]

    train_label_names = [f"label_{i}" for i in range(n_filters)]
    train_label_types = ["int32" for i in range(n_filters)]
    print(f"train_labels_names: {train_label_names}")
    print(f"train_labels_types: {train_label_types}")

    train_label_min = 0
    train_label_max = max_labels

    # label 的设计默认将不同的label限制在不同的数量级中
    range_set = [1]
    for i in range(n_filters):
        if range_set[i] >= train_label_max/1000: 
            for j in range(n_filters-i-1):
                range_set.append(1)
            break
        range_set.append(range_set[i]*10)
    print(range_set)

    train_label = []
    for i in range(n_filters):
        train_label_max_now = train_label_max/range_set[i]
        train_label.append(np.random.randint(train_label_min, train_label_max_now, size=n_samples-m_test))
    train_labels = np.array(train_label)
    
    train_labels = train_labels.T
    print(train_labels.shape)
    
    test_label_range_names = [[f"label_l_{i}",f"label_r_{i}"] for i in range(n_filters)]
    test_label_range_types = [["int32","int32"] for i in range(n_filters)]  
    print(f"test_label_range_names: {test_label_range_names}")
    print(f"test_label_range_types: {test_label_range_types}")
    
    filter_expr = " and ".join(
        [
            f"{train_label_names[i]} >= " + "{" + f"{label_range[0]}" + "}" + " and "
            f"{train_label_names[i]} <= " + "{" + f"{label_range[1]}" + "}"
            for i,label_range in enumerate(test_label_range_names)
        ]
    )
    print(f"filter_expr: {filter_expr}")

    if n_filters > 1:
        # filter_expr_func =  '''def filter_expr(label_l_0, label_r_0, label_l_1, label_r_1, label_l_2, label_r_2, label_l_3, label_r_3, label_l_4, label_r_4, label_l_5, label_r_5):
        #         return f"label_0 >= {label_l_0} and label_0 <= {label_r_0} and label_1 >= {label_l_1} and label_1 <= {label_r_1} and label_2 >= {label_l_2} and label_2 <= {label_r_2} and label_3 >= {label_l_3} and label_3 <= {label_r_3} and label_4 >= {label_l_4} and label_4 <= {label_r_4} and label_5 >= {label_l_5} and label_5 <= {label_r_5}"  
        #         '''
        filter_expr_func =  '''def filter_expr(label_l_0, label_r_0, label_l_1, label_r_1):
                return f"label_0 >= {label_l_0} and label_0 <= {label_r_0} and label_1 >= {label_l_1} and label_1 <= {label_r_1}"  
                '''
    else:
        filter_expr_func =  '''def filter_expr(label_l_0, label_r_0):
                return f"label_0 >= {label_l_0} and label_0 <= {label_r_0}"  
                '''

    print(f"filter_expr_func: {filter_expr_func}")
    test_label_min = 0
    test_label_max = max_labels
    test_labels_range = generate_random_ranges(generate_type, m_test, n_filters, test_label_min, test_label_max,train_labels,ratio_request)

    write_filter_output(
        out_fn,
        train_X,
        test_X,
        train_labels,
        test_labels_range,
        distance,
        filter_expr_func,
        train_label_names,
        train_label_types,
        test_label_range_names,
        test_label_range_types,
        "float",
        topk,
        ratio_request,
    )


if __name__ == "__main__":

    args = parse_arguments()

    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists("data/" + args.path):
        os.mkdir("data/" + args.path)

    out_fn = "data/" + args.path + f"/artificial-{args.type}-{args.d}d-{args.l}l-{args.a}a-{args.metric}-10{str(args.n).count('0')}.hdf5"

    create_filter(
        out_fn,
        args.type,
        args.d,
        args.n,
        args.m,
        args.center,
        args.l,
        args.maxlabel,
        args.a/100,
        args.metric,
        args.topk,
    )

    # out_ground_truth_fn = args.path + f"{args.type}-{args.d}d-{args.l}l-{args.a}a-groundtruth.hdf5"

    # with h5py.File(out_fn, "r") as f:
    #     train_vec=f["train_vec"][:]
    #     test_vec=f["test_vec"][:]
    #     train_label=f["train_label"][:]
    #     test_label=f["test_label"][:]
    #     write_groundtruth_output(
    #         out_ground_truth_fn,
    #         train_vec,
    #         test_vec,
    #         train_label,
    #         test_label,
    #         args.topk,
    #         ratio_request,
    #     )
    #     f.close()



