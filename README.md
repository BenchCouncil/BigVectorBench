# BigVectorBench

[![LICENSE](https://img.shields.io/github/license/BenchCouncil/BigVectorBench.svg)](https://github.com/BenchCouncil/BigVectorBench/blob/master/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/BenchCouncil/BigVectorBench/bvb-run.yml?branch=main)](https://github.com/BenchCouncil/BigVectorBench/actions/workflows/bvb-run.yml)
[![Issues](https://img.shields.io/github/issues/BenchCouncil/BigVectorBench.svg)](https://github.com/BenchCouncil/BigVectorBench/issues)
[![Issues](https://img.shields.io/github/issues-closed/BenchCouncil/BigVectorBench.svg)](https://github.com/BenchCouncil/BigVectorBench/issues)
[![PR](https://img.shields.io/github/issues-pr/BenchCouncil/BigVectorBench.svg)](<[https://github.com/BenchCouncil/BigVectorBench/issues](https://github.com/BenchCouncil/BigVectorBench/pulls)>)
[![PR](https://img.shields.io/github/issues-pr-closed/BenchCouncil/BigVectorBench.svg)](<[https://github.com/BenchCouncil/BigVectorBench/issues](https://github.com/BenchCouncil/BigVectorBench/pulls)>)

BigVectorBench is an innovative benchmark suite crafted to thoroughly evaluate the performance of vector databases. This project is born out of the realization that existing benchmarks fall short in assessing the critical capabilities of vector databases, particularly in handling heterogeneous data embeddings and executing compound queries. Our suite aims to fill this evaluation gap, providing a comprehensive framework for measuring the efficiency and capacity of vector databases in real-world scenarios.

## Install

### Clone the repository

```bash
git clone https://github.com/BenchCouncil/BigVectorBench.git
```

### Make Environment

Tested on: Ubuntu 20.04

- Docker Engine 27.x

  - For installation instructions, see [Install Docker Engine](https://docs.docker.com/engine/install/).
  - Enable Nvidia GPU support for Docker, see [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Optional, only for GPU-based algorithms).

- Python 3.10 and dependencies
  - Create a new conda environment and install dependencies from `requirements.txt`:

```bash
conda create -n bigvectorbench python=3.10
conda activate bigvectorbench
pip3 install -r requirements.txt
```

### Build Docker Images for Databases

The command below will build the Docker images for all databases/algorithms used in the BigVectorBench.

```bash
python install.py
```

Arguments:

- `--proc NUM`: the number of process to build docker images (default: 1, build serially)
- `--algorithm NAME`: build only the named algorithm image (default: None, build all)

Issues:

- If you encounter an error while building the Docker image due to unreachable URLs, please try to modify the `Dockerfile` to use mirrors for the installation of the required packages.

### Run Benchmarks

```bash
python run.py
```

Arguments:

- `--dataset NAME`: run the benchmark on the named dataset (default: glove-100-angular)
- `-k COUNT`, `--count COUNT`: the number of near neighbours to search for (default: 10)
- `--runs COUNT`: run each algorithm instance COUNT times and use only the best result (default: 5)
- `--algorithm NAME`: run only the named algorithm (default: None, run all)
- `--cpuset-cpus CPUSET`: the CPUs in which to allow the container to run (e.g., 0-2, 0,1) only active in Docker mode (default: None, run on all CPUs)
- `--memory MEMORY`: the memory limit for Docker containers, default is all available memory (default: None, run with all available memory)

Example:

```bash
python run.py --dataset app_reviews-384-euclidean-filter --count 100 --runs 3 --algorithm milvus-hnsw --cpuset-cpus 0-15 --memory 64g
```

## Supported Databases and Algorithms

- [Milvus](https://milvus.io/)
  - milvus-flat
  - milvus-ivfflat
  - milvus-ivfsq8
  - milvus-ivfpq
  - milvus-hnsw
  - milvus-scann
  - milvus-gpu-bf
  - milvus-gpu-ivfflat
  - milvus-gpu-ivfpq
  - milvus-gpu-cagra
- [Weaviate](https://weaviate.io/)
  - weaviate-flat
  - weaviate-hnsw
- [Qdrant](https://qdrant.com/)
  - qdrant
- [Vearch](https://vearch.github.io/)
  - vearch-flat
  - vearch-biivf
  - vearch-ivfflat
  - vearch-ivfpq
  - vearch-hnsw
- [Redis](https://redis.io/)
  - redis-flat
  - redis-hnsw
- [Elasitcsearch](https://www.elastic.co/)
  - elasticsearch-hnsw

**TODO**

- [Vespa](https://vespa.ai/)
- [SPTAG](https://github.com/microsoft/SPTAG)
- [pgvector](https://github.com/pgvector/pgvector)

## Datasets

The datasets are available at [link](https://huggingface.co/datasets/Patrickcode/BigVectorBench/tree/main). The datasets are stored in the HDF5 format. The datasets are divided into the following categories:

### Filter-ann Datasets

| Dataset                          | Data / Query Points | Labels | Embedding Model                                                                     | Dimension | Distance  | Download                                                                                                                                                                                                                                      | Raw Data                                                           |
| -------------------------------- | ------------------- | ------ | ----------------------------------------------------------------------------------- | --------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| ag_news-384-euclidean-filter     | 120,000 / 7,600     | 1      | [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384       | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/ag_news-384-euclidean-filter.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/ag_news-384-euclidean-filter.hdf5)         | [ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)        |
| cc_news-384-euclidean-filter     | 620,643 / 10,000    | 1      | [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384       | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/cc_news-384-euclidean-filter.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/cc_news-384-euclidean-filter.hdf5)         | [cc_news](https://huggingface.co/datasets/vblagoje/cc_news)        |
| app_reviews-384-euclidean-filter | 277,936 / 10,000    | 3      | [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384       | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/app_reviews-384-euclidean-filter.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/app_reviews-384-euclidean-filter.hdf5) | [app_reviews](https://huggingface.co/datasets/sealuzh/app_reviews) |
| amazon-384-euclidean-5filter     | 15,928,208 / 10,000 | 5      | [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384       | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/amazon-384-euclidean-5filter.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/amazon-384-euclidean-5filter.hdf5)         | [amazon](https://nijianmo.github.io/amazon)                        |

### Multi-Modal Datasets

| Dataset                           | Data / Query Points | Modal | Embedding Model                                            | Dimension | Distance  | Download                                                                                                                                                                                                                                        | Raw Data                                                                                |
| --------------------------------- | ------------------- | ----- | ---------------------------------------------------------- | --------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| img-wikipedia-1024-euclidean-mm   | 479,116 / 10,000    | image | [ImageBind](https://github.com/facebookresearch/ImageBind) | 1024      | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/img-wikipedia-1024-euclidean-mm.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/img-wikipedia-1024-euclidean-mm.hdf5)     | [img-wikipedia](https://huggingface.co/datasets/israfelsr/img-wikipedia-simple)         |
| librispeech_asr-1024-euclidean-mm | 104,014 / 2,620     | audio | [ImageBind](https://github.com/facebookresearch/ImageBind) | 1024      | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/librispeech_asr-1024-euclidean-mm.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/librispeech_asr-1024-euclidean-mm.hdf5) | [librispeech_asr](https://huggingface.co/datasets/librispeech_asr)                      |
| gpt4vision-1024-euclidean-mm      | 207,868 / 10,000    | image | [ImageBind](https://github.com/facebookresearch/ImageBind) | 1024      | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/gpt4vision-1024-euclidean-mm.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/gpt4vision-1024-euclidean-mm.hdf5)           | [gpt4vision](https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS) |

### Multi-Vector Datasets

| Dataset                | Data / Query Points | Vectors | Embedding Model                                                             | Dimension | Distance  | Download                                                                                                                                                                                                                  | Raw Data                                                       |
| ---------------------- | ------------------- | ------- | --------------------------------------------------------------------------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| webvid-4-512-euclidean | 1,000,000 / 10,000  | 4       | [CLIP-ViT-B-16](https://huggingface.co/sentence-transformers/clip-ViT-B-16) | 512       | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/webvid-4-512-euclidean.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/webvid-4-512-euclidean.hdf5) | [webvid](https://huggingface.co/datasets/TempoFunk/webvid-10M) |

### Large-Scale Datasets

| Dataset                                                              | Data / Query Points | Embedding Model                                                                                                                 | Dimension | Distance  | Download                                                                                                                                                                                                                                                                                                              | Raw Data                                                                |
| -------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| dbpedia-entities-openai3-text-embedding-3-large-1536-1000k-euclidean | 990,000 / 10,000    | [OpenAI text-embedding-3-large](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) | 1536      | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/dbpedia-entities-openai3-text-embedding-3-large-1536-1000k-euclidean.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/dbpedia-entities-openai3-text-embedding-3-large-1536-1000k-euclidean.hdf5) | [dbpedia-entities](https://huggingface.co/datasets/BeIR/dbpedia-entity) |
| dbpedia-entities-openai3-text-embedding-3-large-3072-1000k-euclidean | 990,000 / 10,000    | [OpenAI text-embedding-3-large](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) | 3072      | Euclidean | [link1](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/dbpedia-entities-openai3-text-embedding-3-large-3072-1000k-euclidean.hdf5), [link2](https://hf-mirror.com/datasets/Patrickcode/BigVectorBench/resolve/main/dbpedia-entities-openai3-text-embedding-3-large-3072-1000k-euclidean.hdf5) | [dbpedia-entities](https://huggingface.co/datasets/BeIR/dbpedia-entity) |

## Let's Work Together

For the development of BigVectorBench, we welcome contributions from the community. If you are interested in contributing to this project, please follow the guidelines below.

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

## Results (Until July 2024)

Figures are available at [link](https://github.com/cococo2000/BigVectorBench/tree/main/results/vldb2025).

## Acknowledgements

## Citation
