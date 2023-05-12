import numpy as np

class Logisticregression:
    def __init__(self, learning_rate=0.1, num_iterations=3000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize parameters
        m, n = X.shape
        self.theta = np.zeros(n + 1)
        X = np.hstack((np.ones((m, 1)), X))

        # Gradient descent
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        predictions = (h >= 0.5).astype(int)
        return predictions
    def predict_probability(self, X):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        probs = np.hstack((1-h.reshape(-1, 1), h.reshape(-1, 1)))
        return probs
