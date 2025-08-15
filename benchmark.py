import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import load_ground_truth_data, load_prediction, load_logits, synthesize_predictions
from iv2_utils.iv2 import pickle_read, pickle_write
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.optimize import minimize_scalar
from tabulate import tabulate
import pandas as pd
import numpy as np
import pickle
import pandas
import copy
import cv2
import os

import numpy as np

def calculate_num_videos_within_threshold(preds, data, threshold = 8, compare = False):
    score = 0
    offsets = []
    for idx, i_pred in enumerate(preds):
        within_range = False
        off = 10000
        for truth in data[idx][-1]:
            if abs(i_pred - truth) < abs(off):
                off = i_pred - truth

            if abs(i_pred - truth) <= threshold:
                within_range = True

        if within_range:
            score += 1

        offsets.append(off)

    score = score / len(data)

    return score

def find_closest(pred, truths):
    """
    Finds the value in truths closest to pred.
    Returns pred itself if truths is empty, to avoid errors downstream,
    though this might lead to a 0 error contribution which might be undesirable.
    Consider raising an error or returning np.nan if truths can be empty and that's an issue.
    """
    if not truths:
        return pred
    return min(truths, key=lambda x: abs(x - pred))

def calculate_mse(preds_with_offset, data, find_closest_func=find_closest):
    """
    Calculates Mean Squared Error between predictions (already offset) and their closest truths.
    data[idx][-1] is expected to be the list of truth values for the idx-th prediction.
    """
    errors_sq = []
    if len(preds_with_offset) != len(data):
        raise ValueError("preds_with_offset and data must have the same length.")

    for idx, p_offset in enumerate(preds_with_offset):
        truth_peaks = data[idx][-1]
        if not truth_peaks:
            pass

        closest_t = find_closest_func(p_offset, truth_peaks)
        errors_sq.append((p_offset - closest_t)**2)

    if not errors_sq:
        return float('inf')
    return np.mean(errors_sq)


def find_best_offset(preds, data, calculate_mse_func=calculate_mse, bounds=(-30, 30)):
    """
    Finds the best offset using scipy.optimize.minimize_scalar.
    """

    def objective_function(offset_to_test):
        preds_shifted = [p + offset_to_test for p in preds]
        return calculate_mse_func(preds_shifted, data)

    result = minimize_scalar(
        objective_function,
        bounds=bounds,
        method='bounded'
    )

    if result.success:
        return result.x
    else:
        print(f"Optimization with SciPy failed: {result.message}")
        return result.x

def offset(preds, data):
    best_off = find_best_offset(preds, data)
    return [x + best_off for x in preds]

def calculate_closest_mse(preds, data):
    def clamp_error(prediction, truth):
        if truth in range(prediction, prediction + 9):
            return 0
        else:
            return (prediction + 8) - truth if truth > prediction + 8 else prediction - truth

    mse = []
    for idx, i_pred in enumerate(preds):
        peak = find_closest(i_pred, data[idx][-1])
        mse.append(abs(i_pred - peak))

    return mse

# ===== MAE Within # Frames Benchmark =====

# Helper Functions
def process_configurations(raw_configs):
    """
    Processes raw configurations to include a label for MAE benchmark.
    Handles both 2-tuple (model, window) and 3-tuple (model, window, label) formats.
    """
    processed_configs = []
    for config_item in raw_configs:
        if len(config_item) == 2:
            model_name, window_size = config_item
            label = f"{model_name} (w={window_size})"
            processed_configs.append((model_name, window_size, label))
        elif len(config_item) == 3:
            processed_configs.append(config_item)
        else:
            print(f"Warning: Skipping invalid configuration item for MAE benchmark: {config_item}")
    return processed_configs

def mae_load_and_prepare_predictions(configurations, benchmark_name, ground_truth_data):
    """Loads and processes predictions (including offsetting) for all configurations for MAE benchmark."""
    all_prepared_predictions = []
    for model_name, window_size, _ in configurations:
        raw_preds = load_prediction(
            model_name=model_name,
            dataset_name=benchmark_name,
            window_size=window_size
        )
        prepared_preds = offset(raw_preds, ground_truth_data)
        all_prepared_predictions.append(prepared_preds)
    return all_prepared_predictions

def mae_calculate_metrics_for_plotting(all_prepared_predictions, ground_truth_data, max_frames):
    """Calculates MAE within N frames metrics for each configuration across a range of frame thresholds for plotting."""
    plot_metrics_by_config = [[] for _ in all_prepared_predictions]
    for frame_thresh in range(max_frames):
        for idx, single_config_preds in enumerate(all_prepared_predictions):
            # Use the existing calculate_num_videos_within_threshold for the metric
            metric_value = calculate_num_videos_within_threshold(single_config_preds, ground_truth_data, threshold=frame_thresh)
            plot_metrics_by_config[idx].append(metric_value)
    return plot_metrics_by_config

def mae_plot_benchmark_results(plot_metrics_by_config, configurations, benchmark_name, max_frames):
    """Plots the MAE within N frames benchmark results."""
    plt.figure(figsize=(10, 6))
    plot_x_axis_frames = range(max_frames)
    for i, metrics_for_single_config in enumerate(plot_metrics_by_config):
        config_label = configurations[i][2]
        plt.plot(plot_x_axis_frames, metrics_for_single_config, label=config_label)

    plt.ylabel("% of videos where MAE ≤ Frames")
    plt.xlabel('Frames (Threshold)')
    plt.title(f"MAE Within N Frames Benchmark: {benchmark_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.xlim(-5, max_frames - 1) # Start x-axis slightly before 0 for visibility
    plt.show()

def mae_print_benchmark_summary_table(all_prepared_predictions, ground_truth_data, configurations, frame_thresholds_for_table, benchmark_name):
    """
    Prints a summary table of MAE within N frames benchmark results for specific frame thresholds.
    Bolds the highest value in each threshold column.
    """
    print(f"\n===== {benchmark_name} MAE Benchmark Summary =====")

    if not frame_thresholds_for_table:
        print("No frame thresholds specified for the table.")
        return
    if not configurations:
        print("No configurations provided for the table.")
        return
    if not all_prepared_predictions:
        print("No predictions available for the table.")
        return
    if len(all_prepared_predictions) != len(configurations):
         print("Warning: Number of predictions does not match number of configurations. Skipping table.")
         return

    percentage_results = []
    for single_config_preds in all_prepared_predictions:
        config_percentages = []
        for f_thresh in frame_thresholds_for_table:
            metric_value = calculate_num_videos_within_threshold(single_config_preds, ground_truth_data, threshold=f_thresh)
            config_percentages.append(metric_value * 100)
        percentage_results.append(config_percentages)

    if not percentage_results or not percentage_results[0]:
        print("No metric results calculated for the table.")
        return

    max_percentages_per_threshold = []
    percentages_by_threshold = list(zip(*percentage_results))

    for threshold_percentages in percentages_by_threshold:
        numeric_percentages = [p for p in threshold_percentages if isinstance(p, (int, float))]
        if numeric_percentages:
             max_val = max(numeric_percentages)
             max_percentages_per_threshold.append(max_val)
        else:
             max_percentages_per_threshold.append(None)

    table_data = []
    headers = ["Configuration"]
    headers.extend([f"MAE ≤ {f_thresh}f" for f_thresh in frame_thresholds_for_table])

    for config_idx, (_, _, label) in enumerate(configurations):
        row_data = [label]
        config_percentages = percentage_results[config_idx]
        for threshold_idx, percentage_value in enumerate(config_percentages):
            formatted_percentage = f"{percentage_value:.2f}%"
            if threshold_idx < len(max_percentages_per_threshold):
                max_val_for_threshold = max_percentages_per_threshold[threshold_idx]
                if max_val_for_threshold is not None and percentage_value == max_val_for_threshold:
                    row_data.append(f'\033[1m{formatted_percentage}\033[0m')
                else:
                    row_data.append(formatted_percentage)
            else:
                 row_data.append(formatted_percentage)

        table_data.append(row_data)

    if not table_data:
        print("No data to generate table.")
        return

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print("\n" + "=" * 80 + "\n")


# Main Benchmark Runner Function
def run_mae_benchmark(
    raw_model_configurations,
    benchmark_dataset_name,
    table_frame_thresholds,
    max_frames_for_plot,
    benchmark_data_source,
    filter_lambda=None
):
    """
    Runs the MAE Within # Frames benchmark for the given configurations.

    Args:
        raw_model_configurations (list): List of model configurations.
            Each item can be ('model_name', window_size) or ('model_name', window_size, 'custom_label').
        benchmark_dataset_name (str): The name of the benchmark dataset (e.g., "act75").
        table_frame_thresholds (list): List of frame numbers for tabular output.
        max_frames_for_plot (int): Maximum frame threshold for the plot.
        benchmark_data_source (dict): Dictionary containing ground truth data.
        filter_lambda (function, optional): Lambda function that takes ground truth peak indices
            and returns True if video should be included. If None, all videos are included.
    """
    print(f"Starting benchmark for dataset: {benchmark_dataset_name}")

    # 1. Process configurations to include labels
    processed_configurations = process_configurations(raw_model_configurations)
    if not processed_configurations:
        print("No valid configurations to process. Exiting benchmark.")
        return

    # 2. Load ground truth data
    try:
        ground_truths = benchmark_data_source[benchmark_dataset_name]
    except KeyError:
        print(f"Error: Ground truth data not found for benchmark '{benchmark_dataset_name}'.")
        print(f"Available keys in benchmark_data_source: {list(benchmark_data_source.keys())}")
        return

    # 3. Build mask for videos to include
    total_videos = len(ground_truths)
    if filter_lambda is None:
        include_mask = [True] * total_videos
        filtered_ground_truths = ground_truths
    else:
        include_mask = []
        filtered_ground_truths = []
        for video_idx, video_data in enumerate(ground_truths):
            peakest = video_data[-1]  # Ground truth peak indices
            if filter_lambda(peakest):
                include_mask.append(True)
                filtered_ground_truths.append(video_data)
            else:
                include_mask.append(False)

        included_videos = sum(include_mask)
        print(f"\nFiltering applied:")
        print(f"- Total videos: {total_videos}")
        print(f"- Included videos: {included_videos}")
        print(f"- Excluded videos: {total_videos - included_videos}")
        print(f"- Inclusion rate: {included_videos / total_videos:.2%}")

        if included_videos == 0:
            print("No videos selected after filtering. Exiting benchmark.")
            return

    # 4. Load and prepare predictions for all configurations
    print("Loading and preparing predictions...")
    all_prepared_predictions = []
    for model_name, window_size, _ in tqdm(processed_configurations, desc="Processing configs"):
        try:
            raw_preds = load_prediction(
                model_name=model_name,
                dataset_name=benchmark_dataset_name,
                window_size=window_size
            )

            # Filter predictions based on mask
            filtered_raw_preds = [raw_preds[i] for i, included in enumerate(include_mask) if included]
            prepared_preds = offset(filtered_raw_preds, filtered_ground_truths)
            all_prepared_predictions.append(prepared_preds)
        except Exception as e:
            print(f"Error processing predictions for {model_name} (window={window_size}): {e}")
            all_prepared_predictions.append(None)  # Mark failed prediction

    # 5. Calculate metrics for plotting
    print("Calculating metrics for plotting...")
    plot_metrics_by_config = [[] for _ in processed_configurations]
    for frame_thresh in range(max_frames_for_plot):
        for idx, single_config_preds in enumerate(all_prepared_predictions):
            if single_config_preds is None:  # Skip failed predictions
                continue
            metric_value = calculate_num_videos_within_threshold(
                single_config_preds,
                filtered_ground_truths,
                threshold=frame_thresh
            )
            plot_metrics_by_config[idx].append(metric_value)

    # 6. Plot the results
    print("Generating plot...")
    valid_configs = [c for c, preds in zip(processed_configurations, all_prepared_predictions) if preds is not None]
    valid_plot_metrics = [metrics for metrics in plot_metrics_by_config if metrics]

    if valid_configs:
        mae_plot_benchmark_results(
            valid_plot_metrics,
            valid_configs,
            benchmark_dataset_name,
            max_frames_for_plot
        )

    # 7. Print the summary table
    print("Generating summary table...")
    valid_predictions = [p for p in all_prepared_predictions if p is not None]
    mae_print_benchmark_summary_table(
        valid_predictions,
        filtered_ground_truths,
        valid_configs,
        table_frame_thresholds,
        benchmark_dataset_name
    )

    print(f"\nBenchmark for {benchmark_dataset_name} finished.")
    print(f"Evaluated {len(filtered_ground_truths)} videos after filtering.")


def mse_load_and_prepare_predictions(configurations, dataset_name, ground_truth_data):
    """
    Loads and processes predictions (including finding and applying the best offset)
    for the given configurations.
    Returns a list of (label, prepared_predictions) tuples for successful configurations.
    """
    successful_results = []
    for model_name, window_size, label in tqdm(configurations, desc="Loading and offsetting"):
        try:
            raw_preds = load_prediction(
                model_name=model_name,
                dataset_name=dataset_name,
                window_size=window_size
            )

            prepared_preds = offset(raw_preds, ground_truth_data)

            successful_results.append((label, prepared_preds))

        except Exception as e:
            print(f"Error processing predictions for {label}: {e}")

    return successful_results

def mse_calculate_and_store_results(prepared_results, ground_truth_data):
    """
    Calculates MSE for each set of prepared predictions and stores results.
    Returns lists for labels and MSE values suitable for plotting and printing.
    """
    mse_values = []
    plot_labels = []
    console_labels = []

    if not prepared_results:
        return mse_values, plot_labels, console_labels

    max_label_len = max(len(label) for label, _ in prepared_results)

    for label, prepared_preds in prepared_results:
        try:
            mse = calculate_mse(prepared_preds, ground_truth_data)

            mse_values.append(mse)
            plot_labels.append(label)
            console_labels.append(label.ljust(max_label_len))

        except Exception as e:
            print(f"Error calculating MSE for {label}: {e}")

    return mse_values, plot_labels, console_labels

def mse_print_summary(console_labels, mse_values, benchmark_dataset_name):
    """Prints the MSE results to the console using tabulate."""
    print(f"\n===== {benchmark_dataset_name} MSE Benchmark Summary =====")
    if not mse_values:
        print("No successful MSE results to display.")
    else:
        table_data = []
        table_data = [[label, f"{mse:.4f}"] for label, mse in zip(console_labels, mse_values)]

        headers = ["Configuration", "MSE"]

        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


def mse_plot_benchmark_results(plot_labels, mse_values, benchmark_dataset_name):
    """Generates and saves the MSE bar chart."""
    if not plot_labels:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_labels)))
    plt.bar(plot_labels, mse_values, color=colors)

    plt.xlabel('Model/Window Configuration')
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE Comparison ({benchmark_dataset_name})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'plots/mse_{benchmark_dataset_name}_comparison.svg', bbox_inches="tight")

    plt.show()


# Main Benchmark Runner Function

def run_mse_benchmark(
    raw_model_configurations,
    benchmark_dataset_name,
    benchmark_data_source
):
    """
    Runs the MSE benchmark for the given configurations.

    Args:
        raw_model_configurations (list): List of model configurations.
            Each item can be ('model_name', window_size) or ('model_name', window_size, 'custom_label').
        benchmark_dataset_name (str): The name of the benchmark dataset (e.g., "act75").
        benchmark_data_source (dict): Dictionary containing ground truth data,
                                      e.g., benchmark_data_source['act75'] = ground_truths_for_act75.
                                      Assumed to be indexed by dataset name.
    """
    print(f"Starting MSE benchmark for dataset: {benchmark_dataset_name}")

    processed_configurations = process_configurations(raw_model_configurations)
    if not processed_configurations:
        print("No valid configurations to process. Exiting benchmark.")
        return

    try:
        ground_truths = benchmark_data_source[benchmark_dataset_name]
    except KeyError:
        print(f"Error: Ground truth data not found for benchmark '{benchmark_dataset_name}'.")
        print(f"Available keys in benchmark_data_source: {list(benchmark_data_source.keys())}")
        return

    prepared_results = mse_load_and_prepare_predictions(
        processed_configurations,
        benchmark_dataset_name,
        ground_truths
    )

    if not prepared_results:
        print("No predictions were successfully loaded and prepared. Exiting benchmark.")
        return

    mse_values, plot_labels, console_labels = mse_calculate_and_store_results(
        prepared_results,
        ground_truths
    )

    if not mse_values:
         print("No MSE values were successfully calculated. Exiting benchmark.")
         return

    mse_print_summary(console_labels, mse_values, benchmark_dataset_name)

    mse_plot_benchmark_results(plot_labels, mse_values, benchmark_dataset_name)

    print(f"\nMSE benchmark for {benchmark_dataset_name} finished.")
    print(f"Plot saved to plots/mse_{benchmark_dataset_name}_comparison.svg")

# New Helper for Overview Benchmark
#
def calculate_absolute_errors_per_point(predictions, ground_truth_data):
    """
    Calculates the absolute error for each data point (e.g., video)
    between the prediction and the closest ground truth peak for that point.

    Args:
        predictions (list or np.ndarray): List of prediction values.
        ground_truth_data (list): List where each element corresponds to a prediction
                                  and the last item in the element is a list of ground truth peaks.
                                  Expected format: data[idx][-1] is a list of truths for prediction idx.

    Returns:
        list: A list of absolute error values, one for each data point.
              Returns an empty list if inputs are invalid or lengths don't match.
    """
    errors = []
    if len(predictions) != len(ground_truth_data):
        print(f"Prediction count ({len(predictions)}) does not match ground truth count ({len(ground_truth_data)}). Cannot calculate per-point errors.")
        return []

    for idx, prediction in enumerate(predictions):
        truth_peaks = ground_truth_data[idx][-1]

        closest_truth = find_closest(prediction, truth_peaks)

        error = abs(prediction - closest_truth)
        errors.append(error)

    return errors

# Main Overview Benchmark Runner

def run_overview_benchmark(
    raw_model_configurations,
    benchmark_dataset_name,
    benchmark_data_source,
    max_display_points=None
):
    """
    Runs the Line Plot (Absolute Error per point) benchmark for the given configurations.
    Plots the absolute error over the sequence of data points for comparison.

    Args:
        raw_model_configurations (list): List of model configurations.
            Each item can be ('model_name', window_size) or ('model_name', window_size, 'custom_label').
        benchmark_dataset_name (str): The name of the benchmark dataset (e.g., "act75").
        benchmark_data_source (dict): Dictionary containing ground truth data,
                                      e.g., benchmark_data_source['act75'] = ground_truths_for_act75.
                                      Assumed to be indexed by dataset name.
        max_display_points (int, optional): If provided, limits the plot to the
                                            first `max_display_points` data points.
                                            Defaults to None (plot all points).
    """
    print(f"\n===== Starting Overview (Line Plot) Benchmark for dataset: {benchmark_dataset_name} =====")

    processed_configurations = process_configurations(raw_model_configurations)
    if not processed_configurations:
        print("No valid configurations to process. Exiting benchmark.")
        return

    try:
        ground_truths = benchmark_data_source[benchmark_dataset_name]
    except KeyError:
        print(f"Error: Ground truth data not found for benchmark '{benchmark_dataset_name}'.")
        print(f"Available keys in benchmark_data_source: {list(benchmark_data_source.keys())}")
        return

    if max_display_points is not None and max_display_points > 0:
         ground_truths_subset = ground_truths[:max_display_points]
         num_points_to_plot = len(ground_truths_subset)
         print(f"Limiting plot to the first {num_points_to_plot} data points.")
    else:
        ground_truths_subset = ground_truths
        num_points_to_plot = len(ground_truths_subset)
        print(f"Plotting {num_points_to_plot} data points.")

    if not ground_truths_subset:
        print("No ground truth data available for plotting after applying limits. Exiting benchmark.")
        return

    plt.figure(figsize=(12, 7), dpi=100)

    print("Loading predictions, calculating errors, and plotting...")
    successful_configs = []

    for idx, (model_name, window_size, label) in enumerate(tqdm(processed_configurations, desc="Processing configurations")):
        try:
            raw_preds = load_prediction(
                model_name=model_name,
                dataset_name=benchmark_dataset_name,
                window_size=window_size
            )

            if max_display_points is not None and max_display_points > 0:
                raw_preds_subset = raw_preds[:max_display_points]
            else:
                raw_preds_subset = raw_preds

            per_point_errors = calculate_absolute_errors_per_point(
                raw_preds_subset,
                ground_truths_subset
            )

            if not per_point_errors:
                print(f"Skipping plotting for {label} due to empty error list.")
                continue

            plt.plot(
                range(1, num_points_to_plot + 1),
                per_point_errors,
                linewidth=2,
                label=label,
            )
            successful_configs.append(label)

        except Exception as e:
            print(f"Error processing or plotting for {label}: {e}")

    if not successful_configs:
        print("No configurations were successfully processed and plotted. Exiting benchmark.")
        plt.close()
        return

    plt.ylabel("Absolute Error")
    plt.xlabel("Data Point Index (Video Index)")
    plt.title(f"Absolute Error Per Data Point ({benchmark_dataset_name})")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f'plots/overview_{benchmark_dataset_name}_comparison.svg'
    plt.savefig(plot_filename, bbox_inches="tight")

    plt.show()

    print(f"\nOverview (Line Plot) benchmark for {benchmark_dataset_name} finished.")
    print(f"Plot saved to {plot_filename}")
