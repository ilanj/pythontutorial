import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


x = []
y = []

# Generate lot of random integers for train data
x = np.random.randint(0, 1000, (500, 1))
# print (x)

# Generate square for each integer and save in train data
for i in x:
    y.append(i[0] * i[0])

# print(x, y)

# List of Algorithms
models = []
models.append(('LinearRegression', LinearRegression()))
models.append(('LogisticRegression', LogisticRegression(solver = 'lbfgs', max_iter=120)))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SVC', SVC()))


# Loop through models and identify the best model to predict square
for name, model in models:
    model.fit(x, y)
    # x_predict = 7
    y_predict = model.predict(np.array([[2.3]]))
    print(name, y_predict)