import numpy as np

# K means
# 1. init n data points
# 2. randomly init K centroids
# 3. each data point find closest centroid to go
# 4. update centoids based on groups
# 5. loop over 3 and 4

K = 4

data = np.random.randn(100, 2)

centroids = np.random.randn(K, 2)




for i in range(10):
    groups = {k: [] for k in range(K)}
    for dp in data:
        dists = []
        for cent in centroids:
            dist = np.sqrt(np.sum((dp - cent) ** 2))
            dists.append(dist)
        groups[dists.index(min(dists))].append(dp)

    # calculate centroids
    new_centroids = []
    for key in groups:
        new_centroid = np.mean(groups[key], axis=0)
        new_centroids.append(new_centroid)

    centroids = new_centroids
print(groups)
        