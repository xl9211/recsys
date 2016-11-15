# coding=utf-8
u"""
User: xulin
Date: 13-11-22
"""
from sklearn.cluster import KMeans


class Cluster(object):
    def __init__(self):
        self.k_means = KMeans(
            init='k-means++',
            n_clusters=10,
            n_init=10
        )

    def fit(self, matrix):
        self.k_means.fit(matrix)
        return self.k_means.cluster_centers_

