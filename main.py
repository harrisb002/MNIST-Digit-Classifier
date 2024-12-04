# main.py

from feature_extraction import FeatureExtractor
from dataset import DatasetLoader
from models import Perceptron, NeuralNetwork
from utils import normalize_features, one_hot_encode
import numpy as np
import random

def write_results_to_file(output_path, results):
    """
    Writes the results to a formatted output file.

    Params:
    - output_path (str): Path to the output file.
    - results (dict): Dictionary containing results for each part.
    """
    with open(output_path, 'w') as file:
        file.write("Handwritten Digit Classification Results\n")
        file.write("=" * 50 + "\n\n")
        
        for part, data in results.items():
            file.write(f"## {part}\n")
            file.write("-" * 50 + "\n")
            for key, value in data.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
        file.write("=" * 50 + "\n")
        file.write("End of Results\n")

def main():
    output_file = "classification_results.txt"
    results = {}

    # Part 1: Binary Classification for digits 7 and 9
    print("Part 1: Binary Classification for digits 7 and 9")
    # Load data for digits 7 and 9
    train_7 = DatasetLoader.get_images_and_labels(
        DatasetLoader.read_file('./Train/train7.csv'))
    train_9 = DatasetLoader.get_images_and_labels(
        DatasetLoader.read_file('./Train/train9.csv'))
    val_7 = DatasetLoader.get_images_and_labels(
        DatasetLoader.read_file('./Valid/valid7.csv'))
    val_9 = DatasetLoader.get_images_and_labels(
        DatasetLoader.read_file('./Valid/valid9.csv'))

    # Combine data
    train_data = train_7 + train_9
    val_data = val_7 + val_9
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Extract features and labels
    X_train = []
    y_train = []
    for label, img in train_data:
        features = FeatureExtractor.extract_features(img)
        X_train.append(features)
        y_train.append(1 if label == 9 else -1)

    X_val = []
    y_val = []
    for label, img in val_data:
        features = FeatureExtractor.extract_features(img)
        X_val.append(features)
        y_val.append(1 if label == 9 else -1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Normalize features
    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean, std)

    # Train perceptron
    perceptron = Perceptron(input_size=X_train.shape[1], num_classes=2, learning_rate=0.01)
    best_weights, min_error_frac = perceptron.train(X_train, y_train, X_val, y_val, epochs=1000)
    print(f"Minimum Validation Error Fraction: {min_error_frac}")

    # Test perceptron
    test_data = val_data  # For demonstration; replace with actual test data
    X_test = []
    y_test = []
    for label, img in test_data:
        features = FeatureExtractor.extract_features(img)
        X_test.append(features)
        y_test.append(1 if label == 9 else -1)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test, _, _ = normalize_features(X_test, mean, std)

    test_predictions = perceptron.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save results for Part 1
    results["Binary Classification (Part 1)"] = {
        "Minimum Validation Error Fraction": f"{min_error_frac:.4f}",
        "Test Accuracy": f"{test_accuracy * 100:.2f}%",
        "Context": "Binary classification of digits 7 and 9 using perceptron."
    }

    # Part 2: Multiclass Classification for digits 0-9
    print("\nPart 2: Multiclass Classification for digits 0-9")
    # Load training and validation data
    train_dir = './Train'
    val_dir = './Valid'
    train_dataset = DatasetLoader.extract_folder(train_dir)
    val_dataset = DatasetLoader.extract_folder(val_dir)
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)

    # Extract features and labels
    X_train = []
    y_train = []
    for label, img in train_dataset:
        features = FeatureExtractor.extract_features(img)
        X_train.append(features)
        y_train.append(label)

    X_val = []
    y_val = []
    for label, img in val_dataset:
        features = FeatureExtractor.extract_features(img)
        X_val.append(features)
        y_val.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Normalize features
    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean, std)

    # Train multiclass perceptron
    perceptron = Perceptron(input_size=X_train.shape[1], num_classes=10, learning_rate=0.01)
    best_weights, min_error_frac = perceptron.train(X_train, y_train, X_val, y_val, epochs=1000)
    print(f"Minimum Validation Error Fraction: {min_error_frac}")

    # Test perceptron
    test_data = DatasetLoader.get_images_and_labels(
        DatasetLoader.read_file('test1.csv'))  # Replace with actual test data
    X_test = []
    y_test = []
    for label, img in test_data:
        features = FeatureExtractor.extract_features(img)
        X_test.append(features)
        y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test, _, _ = normalize_features(X_test, mean, std)

    test_predictions = perceptron.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save results for Part 2
    results["Multiclass Classification (Part 2)"] = {
        "Minimum Validation Error Fraction": f"{min_error_frac:.4f}",
        "Test Accuracy": f"{test_accuracy * 100:.2f}%",
        "Context": "Multiclass classification of digits 0-9 using perceptron."
    }

    # Part 3: Neural Network with one hidden layer
    print("\nPart 3: Neural Network with one hidden layer")
    # Convert labels to one-hot encoding
    y_train_one_hot = one_hot_encode(y_train, num_classes=10)
    y_val_one_hot = one_hot_encode(y_val, num_classes=10)

    # Initialize neural network
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=100, output_size=10, learning_rate=0.1)
    best_W1, best_W2, best_accuracy = nn.train(X_train, y_train_one_hot, X_val, y_val, epochs=1000, target_accuracy=0.95)
    print(f"Best Validation Accuracy: {best_accuracy}")

    # Test neural network
    X_test = []
    y_test = []
    for label, img in test_data:
        features = FeatureExtractor.extract_features(img)
        X_test.append(features)
        y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test, _, _ = normalize_features(X_test, mean, std)

    test_predictions = nn.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save results for Part 3
    results["Neural Network (Part 3)"] = {
        "Best Validation Accuracy": f"{best_accuracy * 100:.2f}%",
        "Test Accuracy": f"{test_accuracy * 100:.2f}%",
        "Context": "Multiclass classification of digits 0-9 using a neural network with one hidden layer."
    }

    # Write results to output file
    write_results_to_file(output_file, results)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
