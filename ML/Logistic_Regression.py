import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("C:/Users/KIIT/Downloads/cleaned_titanic_data.csv")
print(df.columns)
X=df.drop(columns='Survived',axis=1)
y=df[['Survived']]
print(X)
print(y)
print(df.head())
print(df.shape)

#Imbalance Data
print(df[['Survived']].value_counts())
print(round(len(df[df['Survived']==0])/len(df),2))
print(round(len(df[df['Survived']==1])/len(df),2))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train)
print(X_test)

#Model evaluation
lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred_train=lr.predict(X_train)
y_pred_test=lr.predict(X_test)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Model Training

confusion_matrix(y_train,y_pred_train)
sns.heatmap(confusion_matrix(y_train,y_pred_train),annot='true',fmt='.3g')
plt.title('Confusion matrix for model Evaluation')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(accuracy_score(y_train,y_pred_train))
print(classification_report(y_train,y_pred_train))

# Model Test

confusion_matrix(y_test,y_pred_test)
sns.heatmap(confusion_matrix(y_test,y_pred_test),annot='true',fmt='.3g')
plt.title('Confusion matrix for model Evaluation')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(accuracy_score(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))

