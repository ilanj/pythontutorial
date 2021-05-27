import numpy as np
X = np.array([[-1, -1], [-2, -4], [-4, -6], [1, 2]])
Y = np.array([1, 1, 2, 2])
from sklearn.naive_bayes import GaussianNB
GNBclf = GaussianNB()
GNBclf.fit(X, Y)