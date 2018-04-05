import numpy as np
from scipy import stats

class Point(object):
    def __init__(self, features, label=None):
        self.features = features
        self.label = label

    def dimensionality(self):
        return len(self.features)

    def distance(self, other):
        return np.linalg.norm(self.features - other.features)

    def get_label(self):
        return self.label

class Cluster(object):
    def __init__(self, points):
        self.points = points

    def get_points(self):
        return self.points

    def get_label(self):
        labels = [point.get_label() for point in self.points]
        cluster_label, count = stats.mode(labels)
        return cluster_label[0]

    def get_purity(self):
        labels = [point.get_label() for point in self.points]
        cluster_label, count = stats.mode(labels)
        return len(labels), np.float64(count)

    def get_centroid(self):
        centroid = np.zeros(self.points[0].dimensionality())
        for point in self.points:
            centroid += point.get_features() / len(self.points)
        return Point(centroid)

    def equivalent(self, other):
        if len(self.get_points()) != len(other.get_points()):
            return False
        for i in range(len(self.get_points())):
            if self.get_points()[i].distance(other.get_points()[i]) != 0:
                return False
        return True

class ClusterSet(object):
    def __init__(self):
        self.clusters = []

    def add(self, c):
        if c in self.clusters:
            raise ValueError
        self.clusters.append(c)

    def get_clusters(self):
        return self.clusters[:]

    def get_centroids(self):
        centroids = []
        for cluster in self.clusters:
            centroids.append(cluster.get_centroid())
        return centroids

    def get_score(self):
        total_correct = 0
        total = 0
        for c in self.clusters:
            n, n_correct = c.get_purity()
            total = total + n
            total_correct = total_correct + n_correct

        return total_correct / float(total)

    def num_clusters(self):
        return len(self.clusters)

    def equivalent(self, other):
        if len(self.get_clusters()) != len(other.get_clusters()):
            return False
        for i in range(len(self.get_clusters())):
            if not self.get_clusters()[i].equivalent(other.get_clusters()[i]):
                return False
        return True
