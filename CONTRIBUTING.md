# Contributing to BigVectorBench

Contributions to BigVectorBench are welcome from everyone.

### New Datasets

- Add how to generate the dataset in `./bigvectorbench/datasets.py` and transform it into the hdf5 format.

#### Dataset Format

- HDF5 format:
  - Attributes:
    - `type`: the type of the dataset (default: `ann`)
      - `ann` or `dense`: ann datasets and large-scale datasets
      - `filter-ann`: filter-ann datasets
      - `mm-ann`: multi-modal datasets
      - `mv-ann`: multi-vector datasets
      - `sparse`: sparse datasets
    - `distance`: the distance computation method (must be specified)
      - `euclidean`: Euclidean distance
      - `angular`: Angular distance
      - `hamming`: Hamming distance
      - `jaccard`: Jaccard distance
    - `filter_expr_func`: the filter expression function (only available for the filter-ann datasets)
    - `label_names`: the names of the labels (only available for the filter-ann datasets)
    - `label_types`: the types of the labels (only available for the filter-ann datasets, e.g., `int32`)
  - Datasets:
    - `train`: the training vectors (available except for the filter-ann datasets)
    - `test`: the query vectors (available except for the filter-ann datasets)
    - `train_vec`: the training vectors (only available for the filter-ann datasets)
    - `train_label`: the training labels (only available for the filter-ann datasets)
    - `test_vec`: the query vectors (only available for the filter-ann datasets)
    - `test_label`: the query labels (only available for the filter-ann datasets)
    - `distances`: the ground truth distances between the query vectors and the training vectors
    - `neighbors`: the ground truth neighbors containing the indices of the nearest neighbors
- Store the datasets in the `./data` directory.

##### Example

- Filter-ann dataset (e.g., `cc_news-384-euclidean-filter`):

```text
--------------- Attributes ---------------
['distance', 'filter_expr_func', 'label_names', 'label_types', 'type']
distance :  euclidean
filter_expr_func :
def filter_expr(unixtime):
    unixtime_head = unixtime - 3 * 24 * 60 * 60
    unixtime_tail = unixtime
    return f"unixtime >= {unixtime_head} and unixtime <= {unixtime_tail}"

label_names :  ['unixtime']
label_types :  ['int32']
type :  filter-ann

---------------- Datasets ---------------
['distances', 'neighbors', 'test_label', 'test_vec', 'train_label', 'train_vec']
Dataset: distances
(10000, 100)
float32
-----------------------------------------
Dataset: neighbors
(10000, 100)
int32
-----------------------------------------
Dataset: test_label
(10000, 1)
int32
-----------------------------------------
Dataset: test_vec
(10000, 384)
float32
-----------------------------------------
Dataset: train_label
(620643, 1)
int32
-----------------------------------------
Dataset: train_vec
(620643, 384)
float32
-----------------------------------------
```

- Multi-Modal dataset (e.g., `img-wikipedia-1024-euclidean-mm`):

```text
-------------- Attributes ---------------
['distance', 'type']
distance :  euclidean
type :  mm-ann

---------------- Datasets ---------------
['distances', 'modal_test', 'neighbors', 'test', 'train']
Dataset: distances
(10000, 100)
float32
-----------------------------------------
Dataset: modal_test
(10000, 1024)
float32
-----------------------------------------
Dataset: neighbors
(10000, 100)
int32
-----------------------------------------
Dataset: test
(10000, 1024)
float32
-----------------------------------------
Dataset: train
(479116, 1024)
float32
-----------------------------------------
```

- Multi-Vector dataset (e.g., `webvid-4-512-euclidean`):

```text
--------------- Attributes ---------------
['distance', 'type']
distance :  euclidean
type :  mv-ann

---------------- Datasets ---------------
['distances', 'neighbors', 'test', 'train']
Dataset: distances
(10000, 100)
float32
-----------------------------------------
Dataset: neighbors
(10000, 100)
int32
-----------------------------------------
Dataset: test
(10000, 4, 512)
float32
-----------------------------------------
Dataset: train
(100000, 4, 512)
float32
-----------------------------------------
```

- Ann and Large-Scale datasets (e.g., `dbpedia-entities-openai3-text-embedding-3-large-1536-1000k-euclidean`):

```text
--------------- Attributes ---------------
['dimension', 'distance', 'point_type', 'type']
dimension :  1536
distance :  euclidean
point_type :  float
type :  dense

---------------- Datasets ---------------
['distances', 'neighbors', 'test', 'train']
Dataset: distances
(10000, 100)
float64
-----------------------------------------
Dataset: neighbors
(10000, 100)
int64
-----------------------------------------
Dataset: test
(10000, 1536)
float64
-----------------------------------------
Dataset: train
(990000, 1536)
float64
-----------------------------------------
```

### New Algorithms and Databases

- Create a new directory in `./bigvectorbench/algorithms` for the new algorithm or database.
- Implement the algorithm or the API for the database in `./bigvectorbench/algorithms/<new_algo>/module.py`, which contains the functions to build the index, search for the nearest neighbors, and retrieve the results. Please refer to the existing algorithms or databases for the implementation and inherit the `BaseANN` class.
- Create a Dockerfile in `./bigvectorbench/algorithms/<new_algo>/Dockerfile` to build the Docker image, which contains the runtime of the algorithm or database.
- Create a configuration file in `./bigvectorbench/algorithms/<new_algo>/config.yaml` to specify the parameters for the algorithm or database.
- Add the new algorithm or database to `.github/workflows/bvb-run.yml` to automatically run the benchmark on the new algorithm or database.
