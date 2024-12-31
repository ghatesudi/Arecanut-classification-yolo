"""
Script for Visualizing YOLO Training Metrics Across Multiple Trials

This script processes training results from multiple trials of a YOLO model stored in subdirectories.
It extracts metrics such as train loss, validation loss, and top-1/top-5 accuracy from `results.csv` 
files, combines them into a single dataset, and generates visualizations.

Functionality:
1. Traverse a specified directory to find `results.csv` files in trial folders.
2. Combine data from all trials into a single DataFrame.
3. Generate a single consolidated plot with subplots for:
   - Train Loss vs. Epochs
   - Validation Loss vs. Epochs
   - Top-1 Accuracy vs. Epochs
   - Top-5 Accuracy vs. Epochs

Prerequisites:
- Ensure that the directory structure includes subdirectories with `results.csv` files.
- Required Python libraries: os, pandas, matplotlib, seaborn.

Outputs:
- A combined plot (saved as a PNG) showing training metrics across trials for easy comparison.

Usage:
- Update the `base_path` variable with the path to your training results.
- Update the `plot_path` variable with the desired path for saving the output plot.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(base_path):
    """
    Load results from all `results.csv` files in subdirectories of the base path.

    Parameters:
        base_path (str): Path to the directory containing trial subdirectories.

    Returns:
        pd.DataFrame: Combined DataFrame with results from all trials.
    """
    all_data = []
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            result_file = os.path.join(folder_path, "results.csv")
            if os.path.exists(result_file):
                try:
                    data = pd.read_csv(
                        result_file, 
                        usecols=["epoch", "train/loss", "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5"]
                    )
                    data['trial'] = folder  # Add trial identifier
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {result_file}: {e}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def plot_all_metrics(data, save_path):
    """
    Create a single figure with subplots for all metrics.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        save_path (str): File path to save the combined plot.
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)

    # Metrics and corresponding axes
    metrics = [
        ("train/loss", "Train Loss vs. Epochs", "Train Loss", axes[0, 0]),
        ("val/loss", "Validation Loss vs. Epochs", "Validation Loss", axes[0, 1]),
        ("metrics/accuracy_top1", "Top-1 Accuracy vs. Epochs", "Top-1 Accuracy", axes[1, 0]),
        ("metrics/accuracy_top5", "Top-5 Accuracy vs. Epochs", "Top-5 Accuracy", axes[1, 1]),
    ]

    for y, title, ylabel, ax in metrics:
        sns.lineplot(data=data, x="epoch", y=y, hue="trial", palette="tab10", linewidth=1.5, ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(title="Trial", fontsize=10)

    # Save the combined plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved at {save_path}")

def main():
    base_path = "areca_yolo/runs/classify_yolo11_n/"
    plot_path = "plots/combined_metrics.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Load results
    results_df = load_results(base_path)
    if results_df.empty:
        print("No results.csv files found in the specified directory.")
        return

    # Plot all metrics in a single figure with subplots
    plot_all_metrics(results_df, plot_path)

if __name__ == "__main__":
    main()
