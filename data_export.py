"""
Export the results of the experiments to a CSV file.
"""

import argparse
import csv

from bigvectorbench.datasets import DATASETS, get_dataset
from bigvectorbench.plotting.utils import compute_metrics_all_runs
from bigvectorbench.results import load_all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Path to the output file", required=True)
    parser.add_argument("--recompute", action="store_true", help="Recompute metrics")
    parser.add_argument("--batch", action="store_true", help="Export results in batch mode", default=False)
    args = parser.parse_args()

    datasets = DATASETS.keys()
    dfs = []
    for dataset_name in datasets:
        print("Looking at dataset", dataset_name)
        if len(list(load_all_results(dataset_name, batch_mode=args.batch))) > 0:
            results = load_all_results(dataset_name, batch_mode=args.batch)
            dataset, _ = get_dataset(dataset_name)
            results = compute_metrics_all_runs(dataset, results, args.recompute)
            for res in results:
                res["dataset"] = dataset_name
                dfs.append(res)
    if len(dfs) > 0:
        with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
            names = list(dfs[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()
            for res in dfs:
                writer.writerow(res)
    else:
        print("No results found")
