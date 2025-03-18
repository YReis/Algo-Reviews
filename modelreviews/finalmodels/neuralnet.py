from typing import List, Tuple
import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, hidden_layers: List[int]):
        self.hidden_layers: List[int] = hidden_layers
        self.train_data: pd.DataFrame = train_data
        self.test_data: pd.DataFrame = test_data

        
        self.target_column: str = train_data.columns[-1]

        
        input_size = train_data.shape[1] - 1  
        output_size = 1  

        
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        
        self.weights = [np.random.randn(n_in, n_out) * 0.01 for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.biases = [np.zeros((1, n_out)) for n_out in self.layer_sizes[1:]]

        
        self.is_classification = train_data[self.target_column].nunique() == 2

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the data: extracts features and target, and normalizes input features.
        Returns:
            X_train, y_train, X_test, y_test as numpy arrays
        """
        X_train = self.train_data.drop(columns=[self.target_column])
        y_train = self.train_data[self.target_column]
        X_test = self.test_data.drop(columns=[self.target_column])
        y_test = self.test_data[self.target_column]

        
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

        
        if not self.is_classification:
            y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
            y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

        return X_train.values, y_train.values.reshape(-1, 1), X_test.values, y_test.values.reshape(-1, 1)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return x > 0

    def sigmoid_derivative(self, x):
        """Derivative of Sigmoid."""
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform forward propagation through the network.
        Returns:
            Z_values: List of linear transformations at each layer.
            A_values: List of activations at each layer.
        """
        A = X
        A_values = [A]
        Z_values = []

        for W, B in zip(self.weights, self.biases):
            Z = np.dot(A, W) + B
            if self.is_classification and W.shape[1] == 1:
                A = self.sigmoid(Z)  
            else:
                A = self.relu(Z)  
            Z_values.append(Z)
            A_values.append(A)

        return Z_values, A_values

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.01) -> None:
        """
        Train the neural network using backpropagation and gradient descent.
        """
        m = X.shape[0]

        for epoch in range(epochs):
            Z_values, A_values = self.forward(X)
            A_final = A_values[-1]

            
            if self.is_classification:
                loss = -np.mean(y * np.log(A_final + 1e-8) + (1 - y) * np.log(1 - A_final + 1e-8))
            else:
                loss = np.mean((A_final - y) ** 2)  

            
            dA = A_final - y

            gradients_W = []
            gradients_B = []

            for i in reversed(range(len(self.weights))):
                dZ = dA * self.relu_derivative(A_values[i + 1]) if i != len(self.weights) - 1 else dA
                dW = (1 / m) * np.dot(A_values[i].T, dZ)
                dB = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

                gradients_W.insert(0, dW)
                gradients_B.insert(0, dB)

                dA = np.dot(dZ, self.weights[i].T)

            
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * gradients_W[i]
                self.biases[i] -= learning_rate * gradients_B[i]

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform predictions using the trained model.
        """
        _, A_values = self.forward(X)
        A_final = A_values[-1]

        if self.is_classification:
            return (A_final > 0.5).astype(int)  
        else:
            return A_final  

    def trainanduse(self) -> None:
        """
        Train the model and evaluate it on the test set.
        """
        X_train, y_train, X_test, y_test = self.preprocess()
        self.fit(X_train, y_train, epochs=1000, learning_rate=0.1)
        predictions = self.predict(X_test)

        if self.is_classification:
            accuracy = np.mean(predictions == y_test)
            print(f"Test Accuracy: {accuracy:.2f}")
        else:
            mse = np.mean((predictions - y_test) ** 2)
            print(f"Test MSE: {mse:.4f}")



