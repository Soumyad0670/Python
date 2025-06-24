# - Install/Import relevant libraries and Modules
# - EDA
# - Separate Independent and Dependent Data
# - Split your data into train and test
# - Model Training (Linear Regression)
# - Model Prediction
# - Model Evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Step 1: EDA

df =pd.read_csv("C:/Users/KIIT/Downloads/Housing.csv") 
print(df)

print(df.info())
print(df.shape)
print(df.describe())
X=df['area']
y=df['price']
print(plt.scatter(X,y))
plt.show()
print(sns.pairplot(df,hue='price',height=3.6))
plt.show()

print(df.columns)

# Step 2: Separate Independent and Dependent Data
# Method 1:

print(df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus']])
print(df['price'])

# Method 2:

print(df.iloc[: ,1:])
print(df.iloc[:,:1])

# Method 3:

print(df.drop(columns='price',axis=1))
print(df.iloc[:,:1])

# Step 3: Split your data into train and test
# Split your data into train and test

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr=LinearRegression()
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
print(lr.fit(X_train,y_train))
print(lr.coef_)
coef_df = pd.DataFrame({'Coefficient/Weight': lr.coef_}, index=['area'])
print(coef_df)

# Step 4: Model Prediction

y_pred_train=lr.predict(X_train)
y_pred_test=lr.predict(X_test)
print(X_train[:3])
print(y_pred_train[:3])
print(X_test)
print(y_test)
print(y_pred_test[:3])

# Step 5: Model evaluation

mean_squared_error(y_train, y_pred_train)
np.sqrt(mean_squared_error(y_train, y_pred_train))
mean_absolute_error(y_train, y_pred_train)
r2_score(y_train, y_pred_train)

# Step 6: Training Evaluation

def training_evaluation(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(f'MSE : {mse}')
    print(f'RMSE : {rmse}')
    print(f'MAE : {mae}')
    print(f'R2 Score : {r2}')

training_evaluation(y_train,y_pred_train)
plt.scatter(y_train, y_pred_train, color='r', label='Actual Price')
plt.plot([y_train.min(), y_train.max()], [y_pred_train.min(), y_pred_train.max()], lw=3, color='k', label='Predicted Price')
plt.title('Best Fit Line for Train Data')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()

