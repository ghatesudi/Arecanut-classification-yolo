"""
Script for Generating Composite Metric Visualizations for Final YOLO Training Models

This script processes `results.csv` files from final YOLO training folders to create a composite 
visualization of key metrics (train loss, validation loss, top-1 accuracy, and top-5 accuracy) 
for each folder. The results are smoothed using a Gaussian filter for better interpretability.

Functionality:
1. Traverse a base directory containing subdirectories of YOLO training results.
2. Load metrics from `results.csv` in each folder.
3. Generate a composite figure with subplots for:
   - Train Loss
   - Validation Loss
   - Top-1 Accuracy
   - Top-5 Accuracy
4. Save the plots to the corresponding folders as PNG files.

Prerequisites:
- Ensure that the directory structure includes subdirectories with `results.csv` files.
- Required Python libraries: os, pandas, matplotlib, numpy, scipy.

Outputs:
- A PNG file for each folder containing a composite visualization of training metrics.

Usage:
- Update the `base_dir` variable with the path to your final training folders.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def smooth_curve_gaussian(y, sigma=2):
    """
    Smooths a curve using a Gaussian filter.

    Parameters:
        y (array-like): The data points to smooth.
        sigma (int): The standard deviation for Gaussian kernel.

    Returns:
        numpy.ndarray: Smoothed data points.
    """
    return gaussian_filter1d(y, sigma=sigma)

def process_results_folder(folder_path, output_path, sigma=2):
    """
    Processes a single folder containing a `results.csv` file to generate a composite metric plot.

    Parameters:
        folder_path (str): Path to the folder containing the `results.csv` file.
        output_path (str): Path to save the composite plot.
        sigma (int): Smoothing parameter for the Gaussian filter.

    Returns:
        None
    """
    csv_file = os.path.join(folder_path, "results.csv")
    if not os.path.exists(csv_file):
        print(f"No results.csv found in {folder_path}. Skipping.")
        return

    try:
        data = pd.read_csv(csv_file)
        epochs = data['epoch']
        train_loss = data['train/loss']
        val_loss = data['val/loss']
        accuracy_top1 = data['metrics/accuracy_top1']
        accuracy_top5 = data['metrics/accuracy_top5']

        # Create a composite figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Define metric-specific y-axis limits
        y_axis_limits = {
            "train_loss": (-0.05, 1.1),  # Limits for train loss
            "val_loss": (0, 0.7),        # Limits for validation loss
            "accuracy_top1": (0.775, 1), # Limits for top-1 accuracy
            "accuracy_top5": (0.8, 1),   # Limits for top-5 accuracy
        }

        metrics = [
            (train_loss, "Train Loss", "Loss", "blue", axs[0, 0], y_axis_limits["train_loss"]),
            (val_loss, "Validation Loss", "Loss", "red", axs[0, 1], y_axis_limits["val_loss"]),
            (accuracy_top1, "Top-1 Accuracy", "Accuracy", "orange", axs[1, 0], y_axis_limits["accuracy_top1"]),
            (accuracy_top5, "Top-5 Accuracy", "Accuracy", "green", axs[1, 1], y_axis_limits["accuracy_top5"]),
        ]

        for metric, title, ylabel, color, ax, ylim in metrics:
            smoothed_metric = smooth_curve_gaussian(metric, sigma=sigma)
            ax.plot(epochs, metric, label=title, marker='o', color=color, linewidth=1.5, markersize=3)
            ax.plot(epochs, smoothed_metric, linestyle='--', color=color, alpha=0.5)
            ax.set_title(title, loc='center', fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)  # Apply specific y-axis limits
            ax.grid(True, color='lightgray', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {os.path.basename(folder_path)} to {output_path}")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main():
    """
    Main function to generate composite plots for all training folders.

    Iterates through all subdirectories in the base directory, reads the
    `results.csv` file from each, and generates a composite plot of training
    and validation metrics.
    """
    # Base directory containing the folders
    base_dir = "/home/ubuntu/Bioinfo/areca/areca_yolo/runs/final_models"
    
    # Iterate through all subdirectories in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            output_path = os.path.join(base_dir, f"{folder_name}_composite_metrics.png")
            process_results_folder(folder_path, output_path, sigma=2)

if __name__ == "__main__":
    main()
