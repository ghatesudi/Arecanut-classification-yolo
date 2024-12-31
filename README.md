Arecanut Classification and Grading using YOLO with Hyperparameter Tuning

This project aims to classify and grade arecanuts using YOLO (You Only Look Once), an efficient object detection model, with hyperparameter tuning for improved accuracy. The model can classify arecanuts into different grades based on their visual features.

This project includes two distinct scripts for analyzing and visualizing training and hyperparameter tuning results for the arecanut classification model:

Hyperparameter Tuning Analysis: Generates plots comparing training and validation metrics across multiple trials.

Final Training Metrics Analysis: Creates composite plots for individual training runs, showing key metrics in a single image.

Overview of Scripts

1. Hyperparameter Tuning Analysis

Purpose

Generates comparative plots for:

Training Loss vs. Epochs

Validation Loss vs. Epochs

Top-1 Accuracy vs. Epochs

Top-5 Accuracy vs. Epochs

These plots are grouped by trial, enabling easy comparison of multiple hyperparameter tuning experiments.

Input

Each trial folder under the specified base directory should contain a results.csv file with the following columns:

epoch: Epoch number.

train/loss: Training loss.

val/loss: Validation loss.

metrics/accuracy_top1: Top-1 accuracy.

metrics/accuracy_top5: Top-5 accuracy.

Output

For each trial, PNG files are generated:

Loss vs. Epochs (Training and Validation Loss)

Accuracy vs. Epochs (Top-1 and Top-5 Accuracy)

Usage

Place the hyperparameter_tuning_plot.py script in the project directory.

Set the base directory containing trial folders in the script:

base_path = "/path/to/hyperparameter_trials"

Run the script:

python hyperparameter_tuning_plot.py

The plots will be saved in the plots/ directory under the base path.

2. Final Training Metrics Analysis

Purpose

Generates composite plots for individual training runs, showing:

Training Loss

Validation Loss

Top-1 Accuracy

Top-5 Accuracy

Each metric is displayed in a separate subplot, providing a comprehensive overview of the training process.

Input

Each training folder under the specified base directory should contain a results.csv file with the following columns:

epoch: Epoch number.

train/loss: Training loss.

val/loss: Validation loss.

metrics/accuracy_top1: Top-1 accuracy.

metrics/accuracy_top5: Top-5 accuracy.

Output

For each folder, a composite PNG file is generated containing all four metrics.

##Usage

Place the composite_metrics_plot.py script in the project directory.

Set the base directory containing final training folders in the script:

base_dir = "/path/to/final_models"

Run the script:

python composite_metrics_plot.py

The composite plots will be saved in the respective training folders.

Directory Structure


```
project/
├── runs/
│   ├── hyperparameter_trials/
│   │   ├── trial_1/
│   │   │   ├── results.csv
│   │   ├── trial_2/
│   │   │   ├── results.csv
│   │   ├── plots/
│   │   │   ├── loss_vs_epochs.png
│   │   │   ├── accuracy_vs_epochs.png
│   ├── final_models/
│   │   ├── model_1/
│   │   │   ├── results.csv
│   │   │   ├── model_1_composite_metrics.png
│   │   ├── model_2/
│   │   │   ├── results.csv
│   │   │   ├── model_2_composite_metrics.png
├── hyperparameter_tuning_plot.py
├── composite_metrics_plot.py
└── README.md
```

##Configuration

Smoothing (Composite Metrics Analysis)

Adjust the sigma parameter for Gaussian filter smoothing. A higher value makes the curves smoother, while a lower value sharpens them.

sigma = 2  # Increase for smoother curves, decrease for tighter curves

Y-Axis Limits (Composite Metrics Analysis)

You can customize y-axis ranges in the process_results_folder function:

y_axis_limits = {
    "train_loss": (-0.05, 1.1),
    "val_loss": (0, 0.7),
    "accuracy_top1": (0.775, 1),
    "accuracy_top5": (0.8, 1),
}

##Prerequisites

Ensure that the following Python libraries are installed:

pandas

matplotlib

seaborn

scipy

Install them using:

pip install pandas matplotlib seaborn scipy


###Example Commands

Hyperparameter Tuning Analysis

python hyperparameter_tuning_plot.py

This will generate comparative plots for all trials in the specified base_path directory.

Final Training Metrics Analysis

python composite_metrics_plot.py

This will generate composite plots for all training folders in the specified base_dir directory.


###Example Outputs

Hyperparameter Tuning Plots

Training Loss vs. Epochs:
This plot shows the change in training loss over the course of the training epochs. It helps evaluate how well the model is learning during training.

Validation Loss vs. Epochs:
This plot displays the validation loss over epochs, providing insight into how the model's generalization performance improves (or worsens) on unseen data.

Top-1 Accuracy vs. Epochs:
This plot tracks the model's top-1 accuracy on the validation set throughout the epochs, showing how often the model's most confident prediction is correct.

Top-5 Accuracy vs. Epochs:
This plot shows the top-5 accuracy on the validation set. Top-5 accuracy reflects the proportion of times the correct label is within the model's top 5 predictions.


Final Training Composite Plots
Training Loss:
A composite plot summarizing the training loss across epochs, allowing for an overall view of the training progression.

Validation Loss:
A composite plot summarizing the validation loss across epochs, helping identify if the model is overfitting or underfitting.

Top-1 Accuracy:
Displays the final top-1 accuracy across epochs, providing a comprehensive view of how the model's predictions align with the actual labels.

Top-5 Accuracy:
A composite plot summarizing the top-5 accuracy, which can help determine how well the model performs with a broader range of predictions.



Troubleshooting

Common Issues

results.csv not found: Ensure each folder contains a valid results.csv file.

Verify the folder structure matches the expected format in the directory structure.

Missing columns: Verify the results.csv file includes all required columns (epoch, train/loss, etc.).

Open the CSV file and confirm the headers.

Permission issues: Ensure the script has write permissions to save plots.

Use chmod +w to add write permissions if needed.

Library import errors: Install missing Python libraries using:

pip install pandas matplotlib seaborn scipy

Plots not generated: Check for invalid values (e.g., NaN) in the results.csv file.

Clean the data or ensure proper logging during model training.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contribution

Contributions are welcome! If you find bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

YOLOv8: https://github.com/ultralytics/yolov8
Hyperparameter tuning: https://scikit-optimize.github.io/
