import numpy as np

class LinearRegression:
    def __init__ (self, lr=0.001, n_iters=1000):
        #lr = learning rate and n_iters = number of iterations for gradient descent
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None #Calculated later
        self.bias = None #Calculated later
        
    def fit(self, X, y):
        '''
        This is a convention for machine learning, the fit method is defined and takes the training samples and the labels for them
        Involves training step and gradient descent (different to KNN as KNN does not have a training step)
        '''
        #Init Parameters
        n_samples, n_features = X.shape #In this case it is a tuple (80, 1) showing that there are 80 samples with 1 feature each
        self.weights = np.zeros(n_features) #Initialise the weights, currently 0 but random values can also be used
        '''
        np.zeros returns a new array of given shape and type, filled with zeros
        In this case, n_features is the shape of the array
        '''
        self.bias = 0
        
        #Gradient Descent        
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias #Equivalent to ŷ in readme
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) #Equivalent to derivative of w for the update rule, .T means trasposed
            db = (1/n_samples) * np.sum(y_predicted - y)
            #In the formula, 2 is present but omitted here as it is just a scaling factor
            
            self.weights -= self.lr * dw #Update weights
            self.bias -= self.lr * db #Update bias
            
        
    def predict(self, X):
        #Method to approximate and return value of test samples
        y_predicted = np.dot(X, self.weights) + self.bias #Equivalent to ŷ in readme
        return y_predicted