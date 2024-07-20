# BigVectorBench

[![LICENSE](https://img.shields.io/github/license/cococo2000/BigVectorBench.svg)](https://github.com/cococo2000/BigVectorBench/blob/master/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/cococo2000/BigVectorBench/benchmarks.yml?branch=main)](https://github.com/cococo2000/BigVectorBench/actions/workflows/benchmarks.yml)
[![Issues](https://img.shields.io/github/issues/cococo2000/BigVectorBench.svg)](https://github.com/cococo2000/BigVectorBench/issues)
[![Issues](https://img.shields.io/github/issues-closed/cococo2000/BigVectorBench.svg)](https://github.com/cococo2000/BigVectorBench/issues)
[![PR](https://img.shields.io/github/issues-pr/cococo2000/BigVectorBench.svg)]([https://github.com/cococo2000/BigVectorBench/issues](https://github.com/cococo2000/BigVectorBench/pulls))
[![PR](https://img.shields.io/github/issues-pr-closed/cococo2000/BigVectorBench.svg)]([https://github.com/cococo2000/BigVectorBench/issues](https://github.com/cococo2000/BigVectorBench/pulls))

BigVectorBench is an innovative benchmark suite crafted to thoroughly evaluate the performance of vector databases. This project is born out of the realization that existing benchmarks fall short in assessing the critical capabilities of vector databases, particularly in handling heterogeneous data embeddings and executing compound queries. Our suite aims to fill this evaluation gap, providing a comprehensive framework for measuring the efficiency and capacity of vector databases in real-world scenarios.

## Install

### Clone the repository

```bash
git clone <url>
```

### Make Environment

- Docker

```bash
```

- Python 3.10 and dependencies

```bash
# Create a new conda environment
conda create -n bigvectorbench python=3.10
# Activate the environment
conda activate bigvectorbench
# Install dependencies from requirements.txt
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

## Datasets

### Filter

- ag_news-384-euclidean-filter
- cc_news-384-euclidean-filter
- app_reviews-384-euclidean-filter
- amazon-384-euclidean-5filter

### Multi-Modal

- img-wikipedia-1024-euclidean-mm
- librispeech_asr-1024-euclidean-mm
- gpt4vision-1024-euclidean-mm

### Multi-Vector

- webvid-4-512-euclidean

### Big

- dbpedia-entities-openai3-text-embedding-3-large-1536-1000k-euclidean
- dbpedia-entities-openai3-text-embedding-3-large-3072-1000k-euclidean

## Results

## Acknowledgements
