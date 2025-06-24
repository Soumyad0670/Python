'''
Decision Tree Classifier with Pre-Pruning and Post-Pruning
Pruning in decision trees is a technique used to reduce overfitting and improve model generalization.
It helps in removing unnecessary branches that do not contribute significantly to decision-making. 
There are two main types of pruning: Pre-Pruning and Post-Pruning.
'''

# Importing libraries

import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import warnings 
warnings.filterwarnings('ignore')

# Loading dataset
df = sns.load_dataset('iris')
df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Separating X and y datasets
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Visualizing the decision tree
plt.figure(figsize=(16, 10))
tree.plot_tree(dtree, filled=True)
plt.show()

# Model evaluation
y_pred_train = dtree.predict(X_train)
y_pred_test = dtree.predict(X_test)

# Training evaluation
print("\nTraining Evaluation:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

# Test evaluation
print("\nTest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

# Post-pruning: (limiting tree depth)
dtree = DecisionTreeClassifier(max_depth=2)

# Model training
dtree.fit(X_train, y_train)

# Visualizing the pruned tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dtree, filled=True)
plt.show()

# Model evaluation after pruning
y_pred_train = dtree.predict(X_train)
y_pred_test = dtree.predict(X_test)

# Training evaluation after pruning
print("\nTraining Evaluation (Post-pruning):")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

# Test evaluation after pruning
print("\nTest Evaluation (Post-pruning):")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

# Pre-Pruning: Limiting the complexity of the tree

parameters = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

print(parameters)

dtree = DecisionTreeClassifier(
    max_depth=3,                  # Limit tree depth
    min_samples_split=4,          # Minimum samples to split
    min_samples_leaf=2,           # Minimum samples in a leaf node
    max_leaf_nodes=10,            # Maximum number of leaf nodes
    max_features=3                # Maximum number of features used for splitting
)

grid=GridSearchCV(
    dtree,
    param_grid=parameters,
    cv=5,
    scoring= 'accuracy',
    verbose=3
)

# Model training
grid.fit(X_train,y_train)
dtree.fit(X_train, y_train)

print(grid.best_params_)

# Visualizing the pruned decision tree
plt.figure(figsize=(16, 10))
tree.plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['setosa', 'versicolor', 'virginica'])
plt.show()

# Model evaluation
y_pred_train = dtree.predict(X_train)
y_pred_test = dtree.predict(X_test)

# Training evaluation
print("\nTraining Evaluation (Pre-Pruning):")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

# Test evaluation
print("\nTest Evaluation (Pre-Pruning):")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
