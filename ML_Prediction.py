# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:02:27 2021

@author: Enes Zeybek
"""

# 1.Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2.Data Preprocessing
# 2.1 Data Import
data1 = pd.read_csv("tennis_data.txt")
print(data1)

# Data Preprocessing
# Encoder: Categorical -> Numerical
from sklearn.preprocessing import LabelEncoder
data2 = data1.apply(LabelEncoder().fit_transform)
print(data2)

ec = data2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder = "passthrough")
ec = ct.fit_transform(ec)
print(ec)

weather = pd.DataFrame(data = ec, index = range(14), columns = ["overcast", "rainy", "sunny"])
print(weather)
newdata1 = pd.concat([weather, data1.iloc[:,1:3]], axis = 1)
newdata1 = pd.concat([newdata1, data2.iloc[:,-2:]], axis = 1)
print(newdata1)

newdata2 = pd.concat([weather,data1.iloc[:,1:2]], axis = 1)
newdata2 = pd.concat([newdata2, data2.iloc[:,-2:]], axis = 1)

# Dividing Data for Training and Testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(newdata2, newdata1.iloc[:,-3], test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) 

y_pred = regressor.predict(x_test)
print(y_pred) # Humidity Prediction y_test/y_pred

# Backward Elimination
import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values = newdata2, axis = 1)
X_l = newdata1.iloc[:,[0,1,2,3,5,6]].values 
r_ols = sm.OLS(endog = newdata1.iloc[:,4:5], exog = X_l)
r = r_ols.fit()
print(r.summary())

newdata3 =  pd.concat([weather, data1.iloc[:,1:2]], axis = 1)
newdata3 = pd.concat([newdata3, data2.iloc[:,-1:]], axis = 1)

import statsmodels.api as sm # We're getting rid of the highest p-value -> "windy"
X = np.append(arr = np.ones((14,1)).astype(int), values = newdata2, axis = 1)
X_l = newdata1.iloc[:,[0,1,2,3,6]].values
r_ols = sm.OLS(endog = newdata1.iloc[:,4:5], exog = X_l)
r = r_ols.fit()
print(r.summary())

x_train2 = pd.concat([x_train.iloc[:,:-2], x_train.iloc[:,-1:]], axis = 1)
x_test2 = pd.concat([x_test.iloc[:,:-2], x_test.iloc[:,-1:]], axis = 1)

regressor.fit(x_train2, y_train) 
y_pred2 = regressor.predict(x_test2)
print(y_pred2) # with Backward Elimination, our new prediction is more accurate

# Data Visualization
A = y_test.values 
plt.scatter(A, y_pred2, color = "red")
plt.xlabel("y_test") 
plt.ylabel("y_pred2")
plt.plot(A, regressor.predict(x_test2), color = "blue")
plt.show()
