import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("../../../pandas/files/sq_price.csv")
X = data[["sq.feet"]]
y = data[["price"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

ridge_model = Ridge(alpha=0.001)
linear_model = LinearRegression()
bayesian_model = BayesianRidge()
lasso_model = Lasso(alpha=0.5)
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
logistic_model = LogisticRegression() # predicted values are like categories
gaussian_nb = GaussianNB()
#error rmse - r2
degree = 1
poly_reg_model = make_pipeline(PolynomialFeatures(degree=5),LinearRegression())

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
poly_reg_model.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)
elastic_model.fit(X_train, y_train)
bayesian_model.fit(X_train, y_train)
gaussian_nb.fit(X_train, y_train)

y_pred = gaussian_nb.predict(X_test)
r2_score = linear_model.score(X_test, y_test)
print("linear",linear_model.score(X_test, y_test)*100)
print("ridge",ridge_model.score(X_test, y_test)*100)
print("lasso",lasso_model.score(X_test, y_test)*100)
print("polynomial reg model",poly_reg_model.score(X_test, y_test)*100)
print("logistic  reg",logistic_model.score(X_test, y_test)*100)
print("elastic net",elastic_model.score(X_test, y_test)*100)
print("bayesian",bayesian_model.score(X_test, y_test)*100)
print("gaussian-nb",gaussian_nb.score(X_test, y_test)*100)
# print(r2_score*100,'%')

print(y_pred)
plt.scatter(X, y, s=15)
plt.plot(X_test, y_pred, color = 'r')
# plt.show()