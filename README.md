<p align="center">
<img height="300px" alt="image" src="https://github.com/user-attachments/assets/f959695f-66be-46c0-a765-2989caf68ce2" />
</p>

## Overview
This repository contains a Python-based tool for analyzing and processing photography model predictions, specifically focusing on frame analysis in video content.

## Features
- **Multiple Dataset Support**: 
  - Supports three main datasets:
    - ACT75 (75 videos dataset)
    - BF50 (Benchmark dataset)
    - FLASH (ActivityNet Captions dataset with 1k+ videos)
- **Model Analysis Tools**:
  - Frame prediction analysis
  - Logit data processing
  - Prediction synthesis across different window sizes
  - Model performance evaluation

## Repository Structure
```
photography-model/
├── predictions/       # Model predictions and configuration
├── logits/            # Raw Logits
├── plots/             # Plots of the outputs
├── data/              # The BF50 and ACT75 datasets are stored here
├── results.ipynb      # Main results and analysis notebook
├── utils.py           # Core utility functions
├── benchmark.py       # Code & Functions for benchmarking
└── temp.ipynb         # Development and testing notebook
```

## Core Features
1. **Data Loading and Processing**:
   - Support for JSON and pickle data formats
   - Ground truth data management for model evaluation

2. **Prediction Synthesis**:
   - Window size adjustments for predictions
   - Smoothed prediction calculations

3. **Model Evaluation**:
   - Support for multiple model comparisons
   - Frame-by-frame analysis
   - Performance metrics calculation

## Getting Started
1. Ensure you have Python 3.x installed with Jupyter support
2. Install required dependencies
3. Clone this repository
4. Open `results.ipynb` to view analysis results or run your own experiments

## Contributing
Feel free to submit issues and enhancement requests. Please ensure any pull requests maintain the project's code structure and documentation standards.
