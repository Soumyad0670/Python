import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Sample non_linear data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 64, 81]) #qudratic relationship

#Model training
model=LinearRegression()
model.fit(X,y)

#Model prediction
y_pred=model.predict(X)
print(y_pred) #predicted value
print(y) #actual value

#Model evaluation
mse=mean_squared_error(y,y_pred)
print(mse)

accuracy=r2_score(y,y_pred)
print(accuracy)

#Plot 
plot.scatter(X,y,color='b',edgecolors='pink',marker='o',label='Actual fit')
plot.plot(X,y_pred,label='Linear Fit', marker='o')
plot.legend()
plot.grid()
plot.show()

#Using polynomial features
poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)
print(X_poly)

#Model Training
model=LinearRegression()
model.fit(X_poly,y)

#Model prediction
y_pred_poly=model.predict(X_poly)
print(y_pred_poly)

print(y)#actual output
print(y_pred)#linear predicted output
print(y_pred_poly)#polynomial predicted output

#Plot
plot.scatter(X,y,color='b',edgecolors='pink',label='Actual fit')
plot.plot(X,y_pred_poly,label='polynomial Fit', marker='o')
plot.legend()
plot.grid()
plot.show()

#Model Evaulation
mse=mean_squared_error(y,y_pred_poly)
print(mse)

accuracy=r2_score(y,y_pred_poly)
print(accuracy)

