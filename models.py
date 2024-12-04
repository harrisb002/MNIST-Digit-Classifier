# models.py

import numpy as np
from utils import sigmoid, softmax

class Perceptron:
    """
    Perceptron classifier for binary and multiclass classification.
    """

    def __init__(self, input_size, num_classes=2, learning_rate=0.01):
        """
        Initializes the perceptron.

        Params:
        - input_size (int): Number of features.
        - num_classes (int): Number of classes.
        - learning_rate (float): Learning rate.
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        if num_classes == 2:
            # Binary classification
            self.weights = np.random.uniform(-0.1, 0.1, input_size)
        else:
            # Multiclass classification
            self.weights = np.random.uniform(-0.1, 0.1, (num_classes, input_size))

    def train(self, X_train, y_train, X_val, y_val, epochs=1000):
        """
        Trains the perceptron.

        Params:
        - X_train (numpy array): Training features.
        - y_train (numpy array): Training labels.
        - X_val (numpy array): Validation features.
        - y_val (numpy array): Validation labels.
        - epochs (int): Number of epochs.

        Returns:
        - best_weights (numpy array): Weights with the least validation error.
        - min_error_fraction (float): Minimum validation error fraction.
        """
        best_weights = self.weights.copy()
        min_error_fraction = float('inf')

        for epoch in range(epochs):
            if self.num_classes == 2:
                # Binary classification
                for x, y in zip(X_train, y_train):
                    prediction = np.sign(np.dot(self.weights, x))
                    if prediction != y:
                        self.weights += self.learning_rate * (y - prediction) * x
            else:
                # Multiclass classification
                for x, y in zip(X_train, y_train):
                    scores = np.dot(self.weights, x)
                    prediction = np.argmax(scores)
                    if prediction != y:
                        self.weights[y] += self.learning_rate * x
                        self.weights[prediction] -= self.learning_rate * x
            # Validate
            val_predictions = self.predict(X_val)
            error_fraction = np.mean(val_predictions != y_val)
            if error_fraction < min_error_fraction:
                min_error_fraction = error_fraction
                best_weights = self.weights.copy()
            # Optionally, print progress
            # print(f"Epoch {epoch+1}, Validation Error: {error_fraction}")
        self.weights = best_weights
        return best_weights, min_error_fraction

    def predict(self, X):
        """
        Predicts labels for input data.

        Params:
        - X (numpy array): Input features.

        Returns:
        - numpy array: Predicted labels.
        """
        if self.num_classes == 2:
            predictions = np.sign(np.dot(X, self.weights))
            predictions = np.where(predictions >= 0, 1, -1)
            return predictions
        else:
            scores = np.dot(X, self.weights.T)
            predictions = np.argmax(scores, axis=1)
            return predictions

class NeuralNetwork:
    """
    Neural Network with one hidden layer for multiclass classification.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes the neural network.

        Params:
        - input_size (int): Number of input neurons.
        - hidden_size (int): Number of hidden neurons.
        - output_size (int): Number of output neurons.
        - learning_rate (float): Learning rate.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.W2 = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, target_accuracy=0.95):
        """
        Trains the neural network using backpropagation.

        Params:
        - X_train (numpy array): Training features.
        - y_train (numpy array): One-hot encoded training labels.
        - X_val (numpy array): Validation features.
        - y_val (numpy array): Validation labels.
        - epochs (int): Number of epochs.
        - target_accuracy (float): Target validation accuracy.

        Returns:
        - best_W1, best_W2 (numpy arrays): Best weights.
        - best_accuracy (float): Best validation accuracy.
        """
        best_accuracy = 0
        best_W1 = self.W1.copy()
        best_W2 = self.W2.copy()

        N = X_train.shape[0]
        for epoch in range(epochs):
            for k in range(N):
                x = X_train[k].reshape(-1, 1)
                d = y_train[k].reshape(-1, 1)
                # Forward pass
                v1 = np.dot(self.W1, x)
                y1 = sigmoid(v1)
                v2 = np.dot(self.W2, y1)
                y2 = softmax(v2)
                # Backward pass
                e2 = d - y2
                delta2 = e2
                e1 = np.dot(self.W2.T, delta2)
                delta1 = y1 * (1 - y1) * e1
                # Update weights
                self.W2 += self.learning_rate * np.dot(delta2, y1.T)
                self.W1 += self.learning_rate * np.dot(delta1, x.T)
            # Validate
            accuracy = self.evaluate(X_val, y_val)
            # print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_W1 = self.W1.copy()
                best_W2 = self.W2.copy()
            if best_accuracy >= target_accuracy:
                break
        self.W1 = best_W1
        self.W2 = best_W2
        return best_W1, best_W2, best_accuracy

    def predict(self, X):
        """
        Predicts labels for input data.

        Params:
        - X (numpy array): Input features.

        Returns:
        - numpy array: Predicted labels.
        """
        predictions = []
        for x in X:
            x = x.reshape(-1, 1)
            v1 = np.dot(self.W1, x)
            y1 = sigmoid(v1)
            v2 = np.dot(self.W2, y1)
            y2 = softmax(v2)
            predicted_label = np.argmax(y2)
            predictions.append(predicted_label)
        return np.array(predictions)

    def evaluate(self, X, y_true):
        """
        Evaluates the model on validation data.

        Params:
        - X (numpy array): Validation features.
        - y_true (numpy array): True labels.

        Returns:
        - float: Accuracy.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y_true)
        return accuracy
