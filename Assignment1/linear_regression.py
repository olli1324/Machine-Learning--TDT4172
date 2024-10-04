import numpy as np

class LinearRegression:
    
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
            y_pred = np.dot(X, self.weights) + self.bias
            d_w = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            d_b = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * d_w
            self.bias -= self.lr * d_b

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
