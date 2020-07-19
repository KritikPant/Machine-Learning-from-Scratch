import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None #Calculated later
        self.bias = None #Calculated later
    
    def fit(self, X, y):
        #X is a vector of size m times n where m is the number of samples and n is the number of features for its sample
        #y is a 1d row vector of size m
        
        #Initialise paramaters
        n_samples, n_features = X.shape #Unpacks X.shape tuple into its variables
        self.weights = np.zeros(n_features) #Initialise the weights, currently 0 but random values can also be used
        self.bias = 0
        
        #Gradient Descent (Done differently to linear regression here)
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            #Derivatives
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw #Update weights
            self.bias -= self.lr * db #Update bias
        
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted] #Classifies the samples into 
        return np.array(y_predicted_class)
        
    def _sigmoid(self, x): #Generally, _ is used to indicte private methods
        #Only takes one training sample
        return 1 / (1 + np.exp(-x)) #Calculate the exponential of all elements in the input array.