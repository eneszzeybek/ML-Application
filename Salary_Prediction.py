# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:45:44 2021

@author: Enes Zeybek
"""

# 1.Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

# 2.Data Preprocessing
# 2.1 Data Import
data = pd.read_csv("salary_data.txt")
print(data)

# Data Frame Slicing
x = data.iloc[:,2:3] # Independent Variable
y = data.iloc[:,5:] # The Dependent Variable

# NumPy Array Transformation
X = x.values
Y = y.values

print("Correlation Matrix")
print(data.corr()) # Correlation Matrix Shows the Relation of Data to Each Other

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# P Value Calculation
# import statsmodels.api as sm
print("LINEAR OLS")
model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary()) # We eliminated high P-Values from x => Seniority and Score

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

print("POLYNOMIAL OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

# Support Vector Regression
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = np.ravel(sc2.fit_transform(Y))

from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaled, y_scaled)

print("SUPPORT VECTOR REGRESSION OLS")
model3 = sm.OLS(svr_reg.predict(x_scaled), x_scaled)
print(model3.fit().summary())

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)

print("DECISION TREE OLS")
model4 = sm.OLS(r_dt.predict(X), X)
print(model4.fit().summary())

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X, Y.ravel())

print("RANDOM FOREST OLS")
model5 = sm.OLS(rf_reg.predict(X), X)
print(model5.fit().summary())

# R Square (R^2) Calculation
# R^2 Values
print("------------------------------")
print("Linear Regression R2 Değeri:", r2_score(Y, lin_reg.predict(X)))
print("Polynomial Regression R2 Değeri:", r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))
print("Support Vector Regression R2 Değeri:", r2_score(y_scaled, svr_reg.predict(x_scaled)))
print("Decision Tree R2 Değeri:", r2_score(Y, r_dt.predict(X)))
print("Random Forest R2 Değeri:", r2_score(Y, rf_reg.predict(X)))

# Predictions by using Linear Regression
print("C-Level_Prediction:", lin_reg.predict([[9]])) # Salaries of C-Level: 25000, 15000, 15000
print("Expert_Prediction:", lin_reg.predict([[4]])) # Salaries of Expert: 4000, 3000, 6000

# OLS Regression Results => R-squared (uncentered):  
# As Single Parameter => x = data.iloc[:,2:3] & y = data.iloc[:,5:]

# Linear Regression
# R-squared: 0.942

# Polynomial Regression
# R-squared: 0.759

# Support Vector Regression
# R-squared: 0.770

# Decision Tree
# R-squared: 0.751

# Random Forest 
# R-squared: 0.719

# With 3 Parameters (including Seniority and Score) -> x = data.iloc[:,2:5] & y = data.iloc[:,5:]

# Linear Regression
# R-squared: 0.903

# Polynomial Regression
# R-squared: 0.680

# Support Vector Regression
# R-squared: 0.782

# Decision Tree
# R-squared: 0.679

# Random Forest 
# R-squared: 0.713

# When we look in terms of single parameter R^2 it is more successful