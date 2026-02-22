# Crimper Die Verification (Smart Die Validation)

A computer vision and metric learning pipeline to visually verify if the correct crimper die is installed in a machine. 

This repository provides an end-to-end workflow: from ArUco-guided data collection and quota-enforced dataset management, to ArcFace-style PyTorch model training, and finally a live camera UI that validates dies with a PASS/FAIL/UNCERTAIN 3-state logic. It also supports "open-set" deployment, allowing you to add new dies to the system without retraining the model.

## Features

* **ArUco-Guided Data Collection**: Live capture tool that detects 4 ArUco markers to automatically perspective-warp and crop the die region into a consistent square ROI.
* **Structured Dataset Management**: Import images into a structured tree while enforcing per-session, per-die, and per-variation quotas defined in a YAML plan. Includes SHA256 deduplication.
* **Metric Learning (PyTorch)**: Trains an EfficientNet-B0 backbone using an ArcFace-style classification head. Computes and exports class centroids and FAR (False Accept Rate) thresholds.
* **Live Verification**: OpenCV-based live validation script that compares live frames against known class centroids using cosine similarity, utilizing a temporal majority-vote window for robust predictions.
* **Open-Set Support**: Easily add new, unseen dies to the live system by processing a folder of images to generate a new centroid.

## Repository Structure

* **`data-collection/`**
  * `plan.yaml`: Defines target quotas for data splits, sessions, die sizes, and environmental variations.
  * `capture_aruco_crop.py`: Camera UI to capture and warp the Region of Interest (ROI) using ArUco markers.
  * `dataset-manager.py`: Ingests raw captures, deduplicates them, enforces YAML quotas, and builds an index.
  * `dataset-converter.py`: Converts the managed dataset into a PyTorch `ImageFolder` format.
  * `prune_index_missing_files.py`: Utility to clean up the dataset index if files are deleted.
* **`model-training/`**
  * `train-script.py`: Trains the EfficientNet-B0 model, evaluates Verification metrics (TAR@FAR), and exports `deploy_model.pt` and `centroids.npy`.
* **`live-test/`**
  * `live_test.py`: Live camera UI that runs the 3-state validation logic (PASS/FAIL/UNCERTAIN) against the trained centroids.
  * `live_test_with_extra_centroids.py`: Enhanced live test script that can dynamically load extra open-set centroids at runtime.
  * `pairs_*.json`: Test configuration files pairing "recommended" dies against "installed" dies to test performance.
* **`embedding/`**
  * `embed.py`: Generates a new centroid (`.npy`) for an open-set die from a folder of raw images.

## Installation

This project requires Python 3.12 and relies on standard machine learning and vision libraries. 

Install the required dependencies:
```bash
pip install -r requirements.txt
