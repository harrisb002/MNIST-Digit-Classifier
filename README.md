# Handwritten Digit Classification Using Perceptron and Neural Networks

## Overview

Implements a pipeline for classifying handwritten digits from the MNIST dataset. 

Consists of the following parts:

1. **Binary classification using a perceptron** (digits 7 and 9).
2. **Multiclass classification using a perceptron** (digits 0-9).
3. **Multiclass classification using a neural network with one hidden layer**.

---

## Project Structure

### **Part 1: Binary Classification with Perceptron**
- **Objective**: Distinguish between digits 7 and 9.
- **Features**:
  - Pixel density
  - Symmetry
  - Maximum and average horizontal and vertical intersections
- **Steps**:
  1. Load data for digits 7 and 9.
  2. Extract features and normalize them.
  3. Train a perceptron using the pocket algorithm for up to 1000 epochs.
  4. Output results, including optimal weights and test accuracy.

### **Part 2: Multiclass Classification with Perceptron**
- **Objective**: Classify all digits (0-9).
- **Features**: Same as Part 1.
- **Steps**:
  1. Load data for digits 0-9.
  2. Extract features and normalize them.
  3. Train a perceptron using multiclass classification logic.
  4. Output results, including minimum validation error and test accuracy.

### **Part 3: Multiclass Classification with Neural Network**
- **Objective**: Classify all digits (0-9) using a neural network with one hidden layer.
- **Architecture**:
  - Input layer: Number of features
  - Hidden layer: 100 neurons
  - Output layer: 10 neurons (one for each digit class)
- **Steps**:
  1. Load data for digits 0-9.
  2. Extract features and normalize them.
  3. Train the network using backpropagation with softmax activation.
  4. Output results, including best validation accuracy and test accuracy.

---

## Features and Engineering

This project incorporates a rich set of features for classification:
- **Density**: Average pixel intensity.
- **Symmetry**: Vertical symmetry of the digit.
- **Intersections**: Maximum and average horizontal/vertical intersections.
- **Central Pixel Density**: Density in the central 10x10 region.
- **Aspect Ratio**: Width-to-height ratio of the bounding box.
- **Number of Loops**: Detected loops in the image.
- **Edge Density**: Proportion of edge pixels detected by Sobel filters.
- **Density Per Quadrant**: Pixel density in each quadrant.

---

## File Descriptions

The project is organized into well-defined modules:

1. **`feature_extraction.py`**:
   - Contains the `FeatureExtractor` class for feature engineering.
2. **`dataset.py`**:
   - Provides the `DatasetLoader` class for loading and preprocessing datasets.
3. **`models.py`**:
   - Implements the `Perceptron` and `NeuralNetwork` classes for training and prediction.
4. **`utils.py`**:
   - Utility functions for normalization, one-hot encoding, and activation functions.
5. **`main.py`**:
   - Driver script that runs the pipeline and writes results to the output file.

---

## Data Files Used

### **Binary Classification (Part 1)**:
- `train7.csv`, `train9.csv`
- `valid7.csv`, `valid9.csv`

### **Multiclass Classification (Parts 2 & 3)**:
- `train0.csv` to `train9.csv`
- `valid0.csv` to `valid9.csv`
- `test1.csv`

---

## Exec Instructions

1. **Prepare Env**:
   - Install deps:
     ```bash
     pip install numpy pandas scipy opencv-python
     ```
2. **Run the Program**:
   - Execute the `main.py` script:
     ```bash
     python main.py
     ```

3. **View Results**:
   - Results are saved in `classification_results.txt`.

## Goal

- Demonstrate binary and multiclass classification using perceptron and neural network models.
- Using feature engineering techniques.

--- 