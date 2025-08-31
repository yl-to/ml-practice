import numpy as np

# K means
# 1. init n data points
# 2. randomly init K centroids
# 3. each data point find closest centroid to go
# 4. update centoids based on groups
# 5. loop over 3 and 4

K = 4

data = np.random.randn(100, 2)

        