# Project File Descriptions

This document provides a summary of the core Python files used in this Federated Learning project for COVID-19 detection.

## `centralized.py`
This script runs a standard training loop on a single machine using all available data (or specific client splits) combined. It serves as the baseline to compare against federated learning performance. It supports command-line arguments for choosing the model architecture (ResNet/ConvNeXt), optimizer (SGD/AdamW), and learning rate.

## `client.py`
This file defines the Flower Client class. It acts as the bridge between the federated server and the local training logic. It receives instructions (like global model parameters and hyperparameters) from the server, runs the local training loop on its specific data partition, and returns the updated weights.

## `data_loader.py`
This is the data management engine. It reads images directly from the zip archive to save disk I/O. It handles partitioning the dataset into three non-IID clients based on their source (e.g., BIMCV vs. Stonybrook). Before downsampling was added, it primarily used a WeightedRandomSampler to handle class imbalance during loading.

## `model.py`
This file contains the PyTorch model definitions (ResNet50, ConvNeXt) and the core training/testing logic. It includes the functions to adapt standard RGB models to accept grayscale X-ray inputs, the training loop with backpropagation, and the evaluation loop that calculates loss and accuracy.

## `server_utils.py`
This utility manages the server-side operations. It is responsible for evaluating the global model on the hold-out test set after every round. It also handles checkpointing, saving the "best" model based on the lowest loss, and logging training history (accuracy/loss) to JSON files.

## `simulation.py`
This is the main entry point for the Federated Learning experiments. It uses Flower's simulation engine to run multiple clients virtually on limited hardware (using GPU resource allocation). It allows you to select the aggregation strategy (FedAvg or FedProx) and configure simulation parameters like the number of rounds and proximal mu.

## `utils.py`
A standalone plotting script. It scans the results directory for training logs (history.json) and generates comparative graphs for training and testing accuracy and loss, allowing you to visualize how different experiments performed against each other.
