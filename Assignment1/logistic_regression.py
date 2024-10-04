import numpy as np
import pandas as pd
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) 
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)

            d_w = (1 / n_samples) * np.dot(X.T, (y_pred - y))  
            d_b = (1 / n_samples) * np.sum(y_pred - y)  

            self.weights -= self.lr * d_w
            self.bias -= self.lr * d_b

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = np.where(y_pred <= 0.5, 0, 1)  # Convert probabilities to binary class labels
        return class_pred
