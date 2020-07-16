import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

from KNN import KNN

iris = datasets.load_iris() #Uses iris dataset from sklearn
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

'''
X_train = X training samples
X_test = X test samples

y_train = labels for X training samples
y_test = labels for X testing samples
'''

classifier = KNN(k=5)
classifier.fit(X_train, y_train)
predicitions = classifier.predict(X_test)

accuracy = np.sum(predicitions == y_test) / len(y_test)
print(accuracy)

# Inspect data

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()