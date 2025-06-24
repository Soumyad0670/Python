import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import warnings 
warnings.filterwarnings('ignore')

# Loading Datasets

X, y = make_moons(n_samples=20, noise=0.4, random_state=42)
print(X)
print(X.shape)
print(y)

# Plotting the actual graph

plt.scatter(X[:,0], X[:,1], c = y)
plt.show()

# Scaling

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
print(X_scaled)

# DBSCAN

db= DBSCAN(eps=0.3, min_samples=5)
db.fit(X_scaled)
print(db.labels_)

# Plotting the actual graph

plt.scatter(X[:,0], X[:,1],c = y)
plt.show()

# Plotting the predicted graph after DBSCAN 

plt.scatter(X[:,0], X[:,1], c = db.labels_)
plt.show()