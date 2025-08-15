#!/usr/bin/env python3
"""
Convert Hugging Face FLASH Dataset to Frame-Based JSON Format

This script loads the qingy2024/FLASH-Dataset from Hugging Face,
converts time-based annotations to frame numbers using actual video FPS,
and exports the data in the required JSON format.
"""

import json
import os
import argparse
import cv2
from datasets import load_dataset
import numpy as np


def get_video_fps(video_path):
    """
    Get the frame rate (FPS) of a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        float: Frame rate of the video, or 30.0 if file doesn't exist or can't be read
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}, using default FPS of 30.0")
        return 30.0

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_path}, using default FPS of 30.0")
            return 30.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            print(f"Warning: Invalid FPS ({fps}) for {video_path}, using default 30.0")
            return 30.0

        return fps
    except Exception as e:
        print(f"Error reading FPS from {video_path}: {e}, using default 30.0")
        return 30.0


def convert_time_to_frame(time_seconds, fps):
    """
    Convert time in seconds to frame number.

    Args:
        time_seconds (float): Time in seconds
        fps (float): Frames per second

    Returns:
        int: Frame number (rounded to nearest integer)
    """
    return int(round(time_seconds * fps))


def process_flash_dataset(downloaded_clips_path, output_path="data/FLASH.json"):
    """
    Process the FLASH dataset and convert to frame-based JSON format.

    Args:
        downloaded_clips_path (str): Path to the folder containing downloaded video clips
        output_path (str): Path where to save the output JSON file
    """
    print("Loading FLASH dataset from Hugging Face...")
    dataset = load_dataset("qingy2024/FLASH-Dataset")

    # Use the train split (you can modify this if needed)
    train_data = dataset['train']

    print(f"Processing {len(train_data)} video entries...")

    result = []

    for i, item in enumerate(train_data):
        if i % 100 == 0:
            print(f"Processing item {i}/{len(train_data)}")

        video_filename = item['video']
        video_full_path = os.path.join(downloaded_clips_path, video_filename)

        # Get FPS from the actual video file
        fps = get_video_fps(video_full_path)

        # Parse the peaks string (it's stored as a string that needs to be parsed as JSON)
        peaks = item['peaks']

        # Convert each peak to frame numbers
        processed_peaks = []
        for peak in peaks:
            processed_peak = {
                "build_up": convert_time_to_frame(peak['build_up'], fps),
                "peak_start": convert_time_to_frame(peak['peak_start'], fps),
                "peak_end": convert_time_to_frame(peak['peak_end'], fps),
                "drop_off": convert_time_to_frame(peak['drop_off'], fps),
                "caption": peak['caption']
            }
            processed_peaks.append(processed_peak)

        # Create the entry in the required format
        # Add "data/" prefix to the video path
        video_path_with_prefix = f"data/{video_filename}"
        entry = [video_path_with_prefix, processed_peaks]
        result.append(entry)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(result)} video entries")
    print(f"Output saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert FLASH dataset to frame-based JSON format")
    parser.add_argument(
        "downloaded_clips_path",
        help="Path to the folder containing downloaded video clips"
    )
    parser.add_argument(
        "--output",
        default="data/FLASH.json",
        help="Output JSON file path (default: data/FLASH.json)"
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.downloaded_clips_path):
        print(f"Error: Downloaded clips path does not exist: {args.downloaded_clips_path}")
        return 1

    try:
        process_flash_dataset(args.downloaded_clips_path, args.output)
        return 0
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
