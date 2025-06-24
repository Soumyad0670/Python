
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
df = pd.read_csv('C:/Users/KIIT/Downloads/Melbourne_housing_cleaned.csv')
X=df.drop('Price',axis=1)
y = df['Price']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression
lr=LinearRegression()
print(lr.fit(X_train,y_train))
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
linear_coef_df = pd.DataFrame(lr.coef_, index=X.columns, columns=['Coefficient/slope/Weight'])
print(linear_coef_df)

#Ridge Regression
# Ridge Regression (L2 Regularization)
# Ridge regression adds a penalty term (alpha * sum of squared coefficients) to the linear regression
# cost function to prevent overfitting. The alpha parameter controls the strength of regularization.
# Higher alpha values mean stronger regularization and smaller coefficients.

ridge=Ridge(alpha=1,max_iter=1000,tol=0.1)
print(ridge.fit(X_train,y_train))
print(ridge.score(X_train,y_train))
print(ridge.score(X_test,y_test))
ridge_coef_df = pd.DataFrame(ridge.coef_, index=X.columns, columns=['Coefficient/slope/Weight'])
print(ridge_coef_df)

# Lasso Regression (L1 Regularization)
# Lasso regression adds a penalty term (alpha * sum of absolute coefficients) to the linear regression
# cost function to prevent overfitting. The alpha parameter controls the strength of regularization.
# Higher alpha values mean stronger regularization.
# Lasso can reduce coefficients to exactly zero, effectively performing feature selection.

lasso=Lasso(alpha=50,max_iter=1000,tol=0.1)
print(lasso.fit(X_train,y_train))
print(lasso.score(X_train,y_train))
print(lasso.score(X_test,y_test))
lasso_coef_df = pd.DataFrame(lasso.coef_, index=X.columns, columns=['Coefficient/slope/Weight'])
print(lasso_coef_df)