# Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification, regression, and outlier detection.
# When the data is linearly separable, SVM finds the optimal hyperplane that maximizes the margin between the two classes.
# Hyperplane: A decision boundary that separates different classes.
# Support Vectors: The data points closest to the hyperplane. These points influence the position and orientation of the hyperplane.
# Margin: The distance between the hyperplane and the closest data points (support vectors). SVM aims to maximize this margin.
# Kernel: Determines how the data is mapped into a higher-dimensional space.

# C (Regularization Parameter):
# Large C → less margin, focuses on correctly classifying all training examples.
# Small C→ larger margin, allows more misclassifications for better generalization.
# Gamma (γ): Defines the influence of a single training example.
# Low γ → larger influence, smooth decision boundary.
# High γ → smaller influence, more complex decision boundary.

# Importing libraries
import numpy as n
import pandas as pd
import matplotlib.pyplot as plt                         
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("C:/Users/KIIT/Downloads/cleaned_titanic_data.csv")
print(df.columns)
X=df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender']]
y=df[['Survived']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train)
print(X_test)

# SVM model building
model = SVC(kernel='linear', C=1.0, gamma='scale')
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(y_train)
print(y_pred_train)

#SVM model evaluation
#Training
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:/n", confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train,y_pred_train))

#Test
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:/n", confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test,y_pred_test))

# K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks.
# It is one of the simplest and most intuitive algorithms in machine learning.
# KNN is based on the assumption that similar points are located near each other, making it a distance-based algorithm.

# Load dataset
df = pd.read_csv("C:/Users/KIIT/Downloads/cleaned_titanic_data.csv")
print(df.columns)
X=df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender']]
y=df[['Survived']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# SVM model building
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# KNN model evaluation
#Training
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:/n", confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train,y_pred_train))

#Test
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:/n", confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test,y_pred_test))


