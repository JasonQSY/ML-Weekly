import numpy as np
from clustering_classes import Cluster, ClusterSet, Point

def random_assign(points, k):
    """
    Randomly assign many points. Baseline for clustering algorithms.
    """
    clusters = [Cluster([]) for _ in range(k)]
    for point in points:
        clusters[np.random.randint(k)].points.append(point)
    clusterset = ClusterSet()
    for cluster in clusters:
        clusterset.add(cluster)
    return clusterset


def random_init(points, k):
    ids = np.random.choice(len(points), k, replace=False)
    return [points[i] for i in ids]


def k_means_pp_init(points, k):
    ids = [np.random.randint(k)]
    distances = np.zeros(len(points))
    while len(ids) != k:
        for i in range(len(points)):
            min_dis = float('inf')
            for id in ids:
                dis = points[i].distance(points[id])
                if dis < min_dis:
                    min_dis = dis
            distances[i] = min_dis
        distances = np.power(distances, 2)
        distances /= np.sum(distances)
        new_id = np.random.choice(len(points), p=distances)
        ids.append(new_id)
    return [points[i] for i in ids]


def k_means(points, k, init='random'):
    """
    k-means clustering.
    """
    # initialization
    if init == 'random':
        centroids = random_init(points, k)
    elif init == 'kpp':
        centroids = k_means_pp_init(points, k)
    else:
        raise ValueError

    # kmeans iteration
    clusterset = ClusterSet()
    for centroid in centroids:
        cluster = Cluster([centroid])
        clusterset.add(cluster)
    new_clusterset = ClusterSet()
    while True:
        centroids = clusterset.get_centroids()
        clusters = [Cluster([]) for _ in range(k)]
        for point in points:
            nearest = -1
            min_dis = float('inf')
            for i in range(len(centroids)):
                dis = centroids[i].distance(point)
                if dis < min_dis:
                    min_dis = dis
                    nearest = i
            clusters[nearest].points.append(point)
        for cluster in clusters:
            new_clusterset.add(cluster)

        if clusterset.equivalent(new_clusterset):
            break
        clusterset = new_clusterset
        new_clusterset = ClusterSet()

    return clusterset
