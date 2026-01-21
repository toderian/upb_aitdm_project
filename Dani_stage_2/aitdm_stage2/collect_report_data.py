import os
import json
import torch
import pandas as pd
import numpy as np
from data_loader import COVIDxZipDataset, _get_client_sources, ZIP_FILE_PATH, get_standard_transform

RESULTS_DIR = "results"

def get_experiment_metrics():
    """
    Scans the results directory and extracts key performance metrics from history.json.
    """
    print("\n" + "="*50)
    print("EXPERIMENT PERFORMANCE METRICS")
    print("="*50)
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found.")
        return

    subfolders = [f.name for f in os.scandir(RESULTS_DIR) if f.is_dir()]
    subfolders.sort()

    results_data = []

    for exp_name in subfolders:
        hist_path = os.path.join(RESULTS_DIR, exp_name, "history.json")
        if not os.path.exists(hist_path):
            continue

        with open(hist_path, "r") as f:
            data = json.load(f)

        # Extract vectors
        rounds = data.get("rounds", data.get("epochs", []))
        test_acc = data.get("accuracy", [])
        test_loss = data.get("loss", [])
        train_acc = data.get("train_accuracy", [])
        train_loss = data.get("train_loss", [])

        if not test_acc:
            continue

        # Calculate Stats
        best_acc = max(test_acc)
        best_acc_round = rounds[test_acc.index(best_acc)]
        
        min_loss = min(test_loss)
        min_loss_round = rounds[test_loss.index(min_loss)]
        
        final_acc = test_acc[-1]
        final_loss = test_loss[-1]

        # Calculate Stability (Standard Deviation of last 3 rounds)
        if len(test_acc) >= 3:
            stability = np.std(test_acc[-3:])
        else:
            stability = 0.0

        results_data.append({
            "Experiment": exp_name,
            "Best Acc": best_acc,
            "Best Acc Round": best_acc_round,
            "Min Loss": min_loss,
            "Min Loss Round": min_loss_round,
            "Final Acc": final_acc,
            "Final Loss": final_loss,
            "Stability (StdDev)": stability
        })

    # Print Table
    header = f"{'Experiment':<45} | {'Best Acc':<10} | {'Min Loss':<10} | {'Final Acc':<10} | {'Final Loss':<10}"
    print(header)
    print("-" * len(header))
    
    for row in results_data:
        print(f"{row['Experiment']:<45} | {row['Best Acc']:.4f}     | {row['Min Loss']:.4f}     | {row['Final Acc']:.4f}      | {row['Final Loss']:.4f}")

    return results_data

def get_dataset_stats():
    """
    Instantiates the dataset wrappers to count samples per client.
    """
    print("\n" + "="*50)
    print("DATASET DISTRIBUTION STATISTICS")
    print("="*50)

    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Archive '{ZIP_FILE_PATH}' not found. Cannot calculate dataset stats.")
        return

    # 1. Analyze Clients
    print(f"{'Partition':<15} | {'Sources':<30} | {'Total':<8} | {'Positive':<8} | {'Negative':<8} | {'% Positive':<10}")
    print("-" * 95)

    total_train = 0

    # Federated Clients (0, 1, 2)
    for client_id in [0, 1, 2]:
        sources = _get_client_sources(client_id)
        # Use a dummy transform, we just need the metadata
        ds = COVIDxZipDataset(ZIP_FILE_PATH, "train.txt", source_filter=sources, transform=None)
        
        counts = ds.df['label'].value_counts()
        n_pos = counts.get('positive', 0)
        n_neg = counts.get('negative', 0)
        total = len(ds)
        total_train += total
        
        ratio = (n_pos / total) * 100 if total > 0 else 0
        
        sources_str = ", ".join(sources)
        print(f"Client {client_id:<8} | {sources_str:<30} | {total:<8} | {n_pos:<8} | {n_neg:<8} | {ratio:.1f}%")

    # 2. Analyze Global Test Set
    ds_test = COVIDxZipDataset(ZIP_FILE_PATH, "test.txt", source_filter=None, transform=None)
    counts = ds_test.df['label'].value_counts()
    n_pos = counts.get('positive', 0)
    n_neg = counts.get('negative', 0)
    total = len(ds_test)
    ratio = (n_pos / total) * 100 if total > 0 else 0
    
    print("-" * 95)
    print(f"{'Global Test':<15} | {'ALL':<30} | {total:<8} | {n_pos:<8} | {n_neg:<8} | {ratio:.1f}%")
    print("-" * 95)
    print(f"Total Training Images Available: {total_train}")

if __name__ == "__main__":
    get_dataset_stats()
    get_experiment_metrics()