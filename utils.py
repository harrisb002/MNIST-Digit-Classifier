# utils.py

import numpy as np

def normalize_features(features, mean=None, std=None):
    """
    Normalizes features using mean and standard deviation.

    Params:
    - features (numpy array): Feature array.
    - mean (numpy array): Mean values per feature (optional).
    - std (numpy array): Standard deviation per feature (optional).

    Returns:
    - normalized_features (numpy array): Normalized features.
    - mean (numpy array): Mean values used for normalization.
    - std (numpy array): Standard deviation used for normalization.
    """
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
    normalized = (features - mean) / std
    return normalized, mean, std

def one_hot_encode(labels, num_classes):
    """
    Converts labels to one-hot encoded vectors.

    Params:
    - labels (numpy array): Array of labels.
    - num_classes (int): Number of classes.

    Returns:
    - numpy array: One-hot encoded labels.
    """
    return np.eye(num_classes)[labels]

def softmax(v):
    """
    Applies softmax function to the input vector.

    Params:
    - v (numpy array): Input vector.

    Returns:
    - numpy array: Softmax output.
    """
    exp_v = np.exp(v - np.max(v))  # For numerical stability
    return exp_v / np.sum(exp_v)

def sigmoid(x):
    """
    Applies sigmoid activation function.

    Params:
    - x (numpy array): Input array.

    Returns:
    - numpy array: Sigmoid output.
    """
    return 1.0 / (1.0 + np.exp(-x))
