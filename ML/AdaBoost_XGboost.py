import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)

# Train models
ada_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

# Make predictions
ada_pred = ada_clf.predict(X_test)
xgb_pred = xgb_clf.predict(X_test)

# Print results
print("AdaBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, ada_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, ada_pred))

print("\nXGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

# Feature importance comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('AdaBoost Feature Importance')
plt.bar(range(len(ada_clf.feature_importances_)), ada_clf.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')

plt.subplot(1, 2, 2)
plt.title('XGBoost Feature Importance')
plt.bar(range(len(xgb_clf.feature_importances_)), xgb_clf.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()