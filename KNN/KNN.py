import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3): #K is given a default value of 3. This is the number of nearest neighbours to consider
        self.k = k
        
    def fit(self, X, y): #Fits the training sample(X) and training label (y)
        self.X_train = X #Capital X is used to reresent multiple samples
        self.y_train = y
    
    def predict(self, X): #Predict new samples
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x): #Helper method which only gets one sample
        '''
        In this we want to calculate the distances of all of the points to find the K Nearest Neighbours
        Then we want to look at the labels of the K Nearest Neighbours
        Finally we want to do a 'maturity vote' to choose the most common class label (popularity vote)
        '''
        #Compute Distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
                
        #Get K Nearest Samples, labels
        k_indices = np.argsort(distances)[:self.k] #Find the indeces of the K Nearest Neighbours from the array of distances
        k_nearest_labels = [self.y_train[i] for i in k_indices] #Find the class labels of the nearest neighbours
        
        #Maturity Vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1) #Returns a list of tuples of most common in the form of [(value, repititions)]
        return most_common[0][0]