import numpy as np

class BaseRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None #Calculated later
        self.bias = None #Calculated later
        
    def fit(self, X, y):        
        #Initialise Parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #Gradient Descent        
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw #Update weights
            self.bias -= self.lr * db #Update bias
            
    def _approximation(self, X, w, b):
        raise NotImplementedError
    
    def predict(self, X):
        return self._predict(X, self.weights, self.bias)
    
    def _predict(self, x, w, b):
        raise NotImplementedError
    
class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X,w) + b
    
    def _predict(self, X):
        return np.dot(X, w) + b

class LogisticRegression(BaseRegression):
    def _approximation(self, X, w, b):
        linear_model = np.dot(X,w) + b
        self._sigmoid(linear_model)
        
    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted] 
        return np.array(y_predicted_class)
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))