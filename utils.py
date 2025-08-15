# utils.py
import os
import json
from typing import List, Dict, Any, Optional, Union

from easydict import EasyDict as edict

# Attempt to import json_read from iv2_utils.iv2
# If it's not available, provide a fallback for basic functionality.
try:
    from iv2_utils.iv2 import json_read
except ImportError:
    print(
        "Warning: `from iv2_utils.iv2 import json_read` failed. "
        "Using a fallback json_read. Ensure iv2_utils is correctly installed and in PYTHONPATH."
    )
    def json_read(file_path: str) -> Any:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred reading {file_path}: {e}")
            raise

try:
    from iv2_utils.iv2 import json_write
except ImportError:
    print(
        "Warning: `from iv2_utils.iv2 import json_write` failed. "
        "Using a fallback json_write. Ensure iv2_utils is correctly installed and in PYTHONPATH."
    )
    def json_write(data: Any, file_path: str, indent: Optional[int] = 4, ensure_ascii: bool = False) -> None:
        try:
            output_dir = os.path.dirname(file_path)
            if output_dir: # Check if dir part is not empty
                os.makedirs(output_dir, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        except Exception as e:
            print(f"An unexpected error occurred writing to {file_path}: {e}")
            raise


# Define base paths relative to this file (utils.py)
# This assumes utils.py is at the root of your project directory structure.
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT_DIR, "predictions")
LOGITS_DIR = os.path.join(PROJECT_ROOT_DIR, "logits")

# --- Ground Truth Data Loading ---

def load_ground_truth_data() -> edict:
    """
    Loads ground truth data for ACT75 and BF50 datasets from the 'data/' directory.

    The ground truth JSON files (e.g., ACT75.json, BF50.json) are expected to
    contain a list of entries, where each entry is typically:
    [video_path_str, description_str, list_of_correct_frames_int]

    Returns:
        edict: An EasyDict object. You can access the data like:
               `gt_data = load_ground_truth_data()`
               `act75_list = gt_data.act75`
               `bf50_list = gt_data.bf50`
               If a file is not found or cannot be read, the corresponding key
               (e.g., `gt_data.act75`) will have a value of None.
    """
    data_store = edict()

    # Datasets to load: key in edict -> filename in data/
    datasets_info = {
        "act75": "ACT75.json",
        "bf50": "BF50.json"
    }

    for dataset_key, filename in datasets_info.items():
        file_path = os.path.join(DATA_DIR, filename)
        try:
            data_store[dataset_key] = json_read(file_path)
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {file_path}")
            data_store[dataset_key] = None
        except Exception as e:
            print(f"Error reading ground truth file {file_path}: {e}")
            data_store[dataset_key] = None

    return data_store

# --- Prediction and Logits Helper ---

def _extract_video_id_from_path(video_path_in_gt: str) -> str:
    """
    Helper function to extract video ID from the video path string.
    Example: 'ACT75/1.mp4' -> '1'
             'data/bf50/some_video.mp4' -> 'some_video'
    """
    return os.path.splitext(os.path.basename(video_path_in_gt))[0]

# --- Prediction Loading ---

def load_prediction(model_name: str, dataset_name: str,
                    window_size: Union[str, int]) -> Optional[List[int]]:
    """
    Loads prediction data for a specific video, model, and dataset.
    Predictions are expected to be in 'predictions/MODEL_NAME/DATASET_NAME/VIDEO_ID.json'.
    Each prediction file is expected to contain a list of predicted frame numbers.

    Args:
        model_name (str): The name of the model (e.g., "B14", "6B").
                          This corresponds to a directory name in 'predictions/'.
        dataset_name (str): The name of the dataset (e.g., "act75", "bf50").
                            This corresponds to a subdirectory name.
                            (e.g., if "ACT75" is passed, it's used as "act75" for path).
        window_size (Union[str, int]):
            The window size.

    Returns:
        Optional[List[int]]: A list of predicted frame numbers.
                             Returns None if the prediction file is not found or an error occurs.
    """
    if isinstance(window_size, str) and ('/' in window_size or '\\' in window_size):
        video_id_str = _extract_video_id_from_path(window_size)
    else:
        video_id_str = str(window_size) # Ensure it's a string for filename

    # Standardize dataset_name to lowercase for path construction (e.g., "act75")
    dataset_name_for_path = dataset_name.lower()

    prediction_file_path = os.path.join(PREDICTIONS_DIR, model_name, dataset_name_for_path, f"{video_id_str}.json")

    try:
        prediction_data = json_read(prediction_file_path)
        return prediction_data
    except FileNotFoundError:
        # This is a common case (e.g., model didn't predict for all videos), so a less verbose message or no message might be preferred.
        # print(f"Info: Prediction file not found at {prediction_file_path}")
        return None
    except Exception as e:
        print(f"Error reading prediction file {prediction_file_path}: {e}")
        return None



# --- Logits Loading ---

def load_logits(model_name: str, dataset_name: str) -> Optional[List[List[List[Union[float, int]]]]]:
    """
    Loads logits data for a given model and dataset.
    Logits are expected to be in 'logits/MODEL_NAME/DATASET_NAME.json'.
    The structure is expected to be a list of videos, where each video has a list of
    [logit_value, frame_number] pairs.

    Args:
        model_name (str): The name of the model (e.g., "B14", "6B").
                          This corresponds to a directory name in 'logits/'.
        dataset_name (str): The name of the dataset (e.g., "act75", "bf50").
                            The JSON filename will be '{dataset_name_lowercase}.json'.

    Returns:
        Optional[List[List[List[Union[float, int]]]]]:
            A list of logit data. Each element in the outer list corresponds to a video.
            Each video's data is a list of [logit_value, frame_number] pairs.
            Example: [ [[7.57, 1], [7.92, 2]], ... (for video 1)
                       [[...], [...]], ... (for video 2) ... ]
            Returns None if the file is not found or an error occurs.
    """
    # Standardize dataset_name to lowercase for path construction (e.g., "act75.json")
    dataset_name_for_path = dataset_name.lower()

    logits_file_path = os.path.join(LOGITS_DIR, model_name, f"{dataset_name_for_path}.json")

    try:
        logits_data = json_read(logits_file_path)
        return logits_data
    except FileNotFoundError:
        # print(f"Info: Logits file not found at {logits_file_path}")
        return None
    except Exception as e:
        print(f"Error reading logits file {logits_file_path}: {e}")
        return None

def _calculate_smoothed_prediction_for_video(video_logits: List[List[Union[float, int]]]) -> int:
    """
    Helper: Calculates a single predicted frame for a video based on smoothed logits.
    Falls back to original argmax if smoothing isn't possible or video_logits is too short.
    Returns predicted frame number, or -1 if no logits are available/valid.
    """
    if not video_logits: # No logit entries for this video
        return -1

    # Ensure all logit entries are valid pairs [value, frame]
    valid_video_logits = []
    for entry in video_logits:
        if isinstance(entry, list) and len(entry) == 2 and \
           isinstance(entry[0], (int, float)) and isinstance(entry[1], int):
            valid_video_logits.append(entry)
        # else:
            # print(f"Warning: Invalid logit entry skipped: {entry}") # Optional: for debugging

    if not valid_video_logits: # All entries were invalid or original list was empty
        return -1

    if len(valid_video_logits) < 3:
        # Fallback: argmax of original (valid) logits
        _best_score, predicted_frame = max(valid_video_logits, key=lambda x: x[0])
        return int(predicted_frame)
    else:
        # Perform smoothing: average previous, current, and next logits
        smoothed_video_scores = [] # Store (avg_val, frame_num)
        for k in range(1, len(valid_video_logits) - 1):
            prev_logit_val = valid_video_logits[k-1][0]
            curr_logit_val = valid_video_logits[k][0]
            next_logit_val = valid_video_logits[k+1][0]

            avg_val = (prev_logit_val + curr_logit_val + next_logit_val) / 3.0
            original_frame_num = valid_video_logits[k][1] # Frame of the central logit
            smoothed_video_scores.append((avg_val, original_frame_num))

        if not smoothed_video_scores: # Should only happen if len(valid_video_logits) was < 3
            _best_score, predicted_frame = max(valid_video_logits, key=lambda x: x[0])
            return int(predicted_frame)

        _best_avg_val, predicted_frame = max(smoothed_video_scores, key=lambda x: x[0])
        return int(predicted_frame)


def synthesize_predictions(model_name: str, dataset_name: str,
                           base_window_size: int, new_window_size: int) -> bool:
    """
    Synthesizes new predictions by smoothing existing logits and saves them.

    Smoothing involves averaging 'previous', 'current', 'next' logit values.
    Argmax of smoothed logits determines predicted frame per video.
    Predictions are saved to 'predictions/MODEL/DATASET/{new_window_size}.json'.

    Args:
        model_name (str): Model name (e.g., "B14").
        dataset_name (str): Dataset name (e.g., "act75").
        base_window_size (int): Base window size of loaded logits (for validation).
        new_window_size (int): Target window size label for new predictions
                               (for validation and output filename).

    Returns:
        bool: True if successful, False otherwise.

    Validation: (new_window_size - base_window_size) must be a positive odd integer.
    """
    diff_window_size = new_window_size - base_window_size
    if not (diff_window_size > 0 and diff_window_size % 2 != 0):
        print(f"Error: Invalid window sizes. (new_window_size - base_window_size) "
              f"must be a positive odd integer. Got base={base_window_size}, new={new_window_size} "
              f"(difference={diff_window_size}). Synthesis aborted.")
        return False

    print(f"Synthesizing predictions for model '{model_name}', dataset '{dataset_name}', "
          f"from base window {base_window_size} to new window {new_window_size}.")

    all_logits_data = load_logits(model_name, dataset_name)
    if all_logits_data is None:
        print(f"Failed to load logits for {model_name}/{dataset_name}. Cannot synthesize.")
        return False
    if not all_logits_data:
        print(f"Logits file for {model_name}/{dataset_name} is empty. No predictions to synthesize.")
        return False

    print(f"Loaded {len(all_logits_data)} videos' logits.")

    synthesized_predictions_list = []
    for i, video_logits in enumerate(all_logits_data):
        predicted_frame = _calculate_smoothed_prediction_for_video(video_logits)
        synthesized_predictions_list.append(predicted_frame)

    dataset_name_lower = dataset_name.lower()
    output_filename = f"{new_window_size}.json"
    output_path = os.path.join(PREDICTIONS_DIR, model_name, dataset_name_lower, output_filename)

    print(f"Saving {len(synthesized_predictions_list)} synthesized predictions to: {output_path}")
    try:
        # Saving as a compact list, similar to example "B14/act75/8.json"
        json_write(synthesized_predictions_list, output_path)
        print("Predictions saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving synthesized predictions to {output_path}: {e}")
        return False

# Example usage (you would typically do this in another script after importing these functions):
if __name__ == '__main__':
    print("--- Example Usage of utils.py ---")

    # 1. Load Ground Truth Data
    print("\n1. Loading Ground Truth Data...")
    gt_data_store = load_ground_truth_data()

    if gt_data_store.act75:
        print(f"Loaded ACT75 ground truth. Number of videos: {len(gt_data_store.act75)}")
        # print(f"First video entry in ACT75: {gt_data_store.act75[0]}")
    else:
        print("ACT75.json not found or failed to load.")

    if gt_data_store.bf50:
        print(f"Loaded BF50 ground truth. Number of videos: {len(gt_data_store.bf50)}")
    else:
        print("BF50.json not found or failed to load.")

    # 2. Load a specific prediction
    print("\n2. Loading a specific prediction...")
    model = "B14" # Example model
    dataset = "act75" # Example dataset

    # Try loading prediction for the 8th video in ACT75 (assuming IDs are 1-based from filenames)
    # The ground truth file ACT75.json has paths like "ACT75/8.mp4"
    # We can use the ID "8" or the path "ACT75/8.mp4"

    # Using video ID directly:
    video_id_to_load = "8"
    predicted_frames = load_prediction(model, dataset, video_id_to_load)
    if predicted_frames:
        print(f"Predictions for Model '{model}', Dataset '{dataset}', Video ID '{video_id_to_load}': {predicted_frames[:5]}...")
    else:
        print(f"No predictions found for Model '{model}', Dataset '{dataset}', Video ID '{video_id_to_load}'. (predictions/{model}/{dataset}/{video_id_to_load}.json)")

    # Example using video path from ground truth (if gt_data_store.act75 is loaded)
    if gt_data_store.act75 and len(gt_data_store.act75) > 7: # Check if 8th video exists (0-indexed)
        video_path_from_gt = gt_data_store.act75[7][0] # e.g., "ACT75/8.mp4"
        predicted_frames_from_path = load_prediction(model, dataset, video_path_from_gt)
        if predicted_frames_from_path:
            print(f"Predictions for Video Path '{video_path_from_gt}': {predicted_frames_from_path[:5]}...")
        else:
            extracted_id = _extract_video_id_from_path(video_path_from_gt)
            print(f"No predictions found for Video Path '{video_path_from_gt}'. (Checked predictions/{model}/{dataset}/{extracted_id}.json)")


    # 3. Load all logits for a model and dataset
    print("\n3. Loading all logits for a model and dataset...")
    model_logits = load_logits(model, dataset)
    if model_logits:
        print(f"Logits loaded for Model '{model}', Dataset '{dataset}'. Number of videos in logits: {len(model_logits)}")
        if model_logits: # Check if not empty
            # print(f"Logits for the first video: {model_logits[0][:3]}...") # Print first 3 frame logits for first video
            pass # Avoid printing too much data
    else:
        print(f"No logits file found for Model '{model}', Dataset '{dataset}'. (logits/{model}/{dataset}.json)")

    print("\n--- End of Example Usage ---")
