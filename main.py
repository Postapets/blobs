import numpy as np
import matplotlib.pyplot as plt
from numpy import array, int64

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=1)

print(X.shape)
print(y.shape)

print(X[1:5, 1])
print(y[1:5])

y[:10]

np.unique(y, return_counts=True)

(array([0, 1, 2]), array([34, 33, 33], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()

X2, y2 = make_blobs(centers=2, random_state=1, center_box=(-2.0, 2.0), cluster_std=1.0, shuffle=False)
print(X2.shape)
print(y2.shape)
print(X2[1:5, 1])
print(y2[:10])

np.unique(y2, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X2[:, 0], X2[:, 1], c=y2)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()
