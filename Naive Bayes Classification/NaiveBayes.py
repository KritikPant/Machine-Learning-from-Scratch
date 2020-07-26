import numpy as np

class NaiveBayes:
	#Does not need __init__ method
	def fit(self, X, y):
		n_samples, n_features = X.shape
		self._classes = np.unique(y)
		#Finds unique elements as they are the different classes which data can be sorted into
		n_classes = len(self._classes)

		#Initialise mean, variants and priors
		self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
		self._var = np.zeros((n_classes, n_features), dtype = np.float64)	
		self._priors = np.zeros(n_classes, dtype = np.float64)

		for c in self._classes:
			X_c = X[c==y]
			self._mean[c, :] = X_c.mean(axis=0)
			self._mean[c, :] = X_c.var(axis=0)
			self._priors[c] = X_c.shape[0]/float(n_samples) #Prior probability = frequency in training data

	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return y_pred

	def _predict(self, x):
		#The function which contains the logarithms
		posteriors = []

		for index, c in enumerate(self._classes):
			prior = np.log(self._priors[index])
			class_conditional = np.sum(np.log(self._pdf[index, x]))
			posterior = prior + class_conditional
			posteriors.append(posterior)
		
		return self._classes[np.argmax(posteriors)]

	def _pdf(self, class_index, x): #Probability density function
		mean = self._mean[class_index]
		var = self._var[class_index]
		numerator = np.exp(- (x - mean ** 2 / (2 * var)))
		denominator = np.sqrt(2 * np.pi * var)
		return numerator/denominator
