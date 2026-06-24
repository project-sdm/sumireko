import math

import numpy as np


def dist(a, b):
    sum = 0

    for c1, c2 in zip(a, b):
        sum += (c1 - c2) ** 2

    return math.sqrt(sum)


def kmeans(points: np.ndarray, k: int):
    centroids = points[np.random.choice(len(points), k, replace=False)]

    while True:
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.empty_like(centroids)

        for i in range(k):
            new_centroids[i] = points[labels == i].mean(axis=0)

        if np.allclose(centroids, new_centroids, atol=1e-2):
            return centroids

        centroids = new_centroids
