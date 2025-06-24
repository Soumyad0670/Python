import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import warnings

warnings.filterwarnings('ignore')

# Load dataset
X, y = make_blobs(n_samples=1000, centers=3, cluster_std=0.60, random_state=0)

# Print dataset details
print("Raw Dataset")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X sample:\n", X[:10])  # Print first 10 rows
print("y sample:\n", y[:10])

# Scatter plot of original clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
plt.title("Original Data with True Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print train-test split details
print("Train-Test Split")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("X_train sample:\n", X_train[:10])

# Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Print scaled data
print("Scaled Data")
print("X_train_sc sample:\n", X_train_sc[:10])
print("X_test_sc sample:\n", X_test_sc[:10])

# KMeans clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_train_sc)
y_pred_train = kmeans.predict(X_train_sc)

# Print predicted clusters
print("KMeans Cluster Assignments")
print("Predicted Clusters (Train):", np.unique(y_pred_train))
print("Predicted Cluster Labels (First 10 Train Samples):\n", y_pred_train[:10])

# Subplots for visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual clusters
axes[0].scatter(X_train_sc[:, 0], X_train_sc[:, 1], c=y_train, cmap='viridis', s=50)
axes[0].set_title("Actual Clusters")
axes[0].set_xlabel("Feature 1 (Scaled)")
axes[0].set_ylabel("Feature 2 (Scaled)")

# Predicted clusters
axes[1].scatter(X_train_sc[:, 0], X_train_sc[:, 1], c=y_pred_train, cmap='viridis', s=50)
axes[1].set_title("Predicted Clusters (KMeans)")
axes[1].set_xlabel("Feature 1 (Scaled)")
axes[1].set_ylabel("Feature 2 (Scaled)")

plt.show()

# WCSS / Elbow Method
'''
WCSS is used in the Elbow Method to determine the optimal number of clusters for K-Means.
As the number of clusters increases, WCSS decreases because points are closer to their centroids.
The "elbow point" in the WCSS plot indicates the optimal number of clusters, where adding more clusters does not significantly reduce WCSS.
'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train_sc)
    wcss.append(kmeans.inertia_)

# Print WCSS values
print("WCSS (Elbow Method)")
for i, val in enumerate(wcss, start=1):
    print(f"Clusters: {i}, WCSS: {val:.2f}")

# Plot WCSS Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# KneeLocator to find optimal clusters
k1 = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
print("\nOptimal number of clusters:", k1.elbow)

# Silhouette Score Calculation
'''
Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
Higher Silhouette Score indicates better-defined clusters.
It is commonly used to determine the optimal number of clusters in a dataset.
'''
silhouette_scores = []
for i in range(2, 11):  # Start from 2 (Silhouette Score is undefined for k=1)
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train_sc)
    score = silhouette_score(X_train_sc, kmeans.labels_)
    silhouette_scores.append(score)

# Print silhouette scores
print("Silhouette Scores")
for i, score in enumerate(silhouette_scores, start=2):
    print(f"Clusters: {i}, Silhouette Score: {score:.4f}")

# Plot Silhouette Score
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='g')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.grid()
plt.show()
