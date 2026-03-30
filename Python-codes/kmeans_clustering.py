from sklearn.datasets import make_blobs # data set generation
from sklearn.cluster import KMeans # model fitting
import matplotlib.pyplot as plt # data visualization

# data set
X, _ = make_blobs(n_samples = 500, centers = 5, cluster_std = 0.5, random_state = 42)

# data visualization (original data set)
plt.figure(figsize = (6, 5))
plt.scatter(X[:, 0], X[:, 1], s = 30)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# --- k-Means with different k values ---
for k in [5, 10]:
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # --- cluster visualization ---
    plt.figure(figsize = (6, 5))
    plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis", s = 30)
    plt.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "X", s = 200, edgecolors = "black")
    plt.title(f"k-Means Clustering (k = {k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
