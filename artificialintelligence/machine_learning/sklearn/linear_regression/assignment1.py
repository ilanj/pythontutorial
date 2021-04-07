import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
bmi_life_data=pd.read_csv("bmi.txt")
# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
print(type(bmi_life_model))
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
# laos_life_exp = bmi_life_model.predict(bmi_life_data[['BMI']])
laos_life_exp = bmi_life_model.predict(np.array([[26.73339]]))
print(laos_life_exp)

