import sklearn.cluster
import numpy as np

# returns labels and centers
def cluster_image(image, num_clusters):
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=10, max_iter=1000, tol=1e-2)
    clustering = kmeans.fit(image.reshape(-1, 3))
    clipped_centers = np.clip(clustering.cluster_centers_, 0., 255.).astype(np.uint8)

    return clustering.labels_.reshape(image.shape[:-1]), clipped_centers

def recolour(image, labels):
    recoloured = np.zeros(image.shape)
    for i in range(np.max(labels) + 1):
        mask = (labels == i)
        if np.any(mask):
            recoloured[mask] = image[mask].mean(axis=0, keepdims=True)
    return np.clip(recoloured, 0, 255).astype(np.uint8)

