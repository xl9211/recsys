# coding=utf-8
u"""
User: xulin
Date: 13-11-22
"""
from numpy import shape
from scipy.spatial import distance


class Classifier(object):
    def __init__(self, user_class):
        self.user_class = user_class

    @staticmethod
    def cos_sim(u, v):
        # 值越小越相似
        return distance.cosine(u, v)

    def classify(self, user_array):
        row = shape(self.user_class)[0]

        min_score = 2
        class_index = 0
        for r in range(row):
            score = Classifier.cos_sim(user_array, self.user_class[r, :])
            if min_score > score:
                min_score = score
                class_index = r

        return self.user_class[class_index, :]




