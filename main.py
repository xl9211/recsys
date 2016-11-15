# coding=utf-8
u"""
User: xulin
Date: 13-11-21
"""
from recommendation.metric.pairwise import loglikehood_coefficient
from recommendation.model.sale import Sale
from recommendation.model.session import get_session
from recommendation.recommender.knn.method import ItemBasedRecommender
from recommendation.similarity.basic_similarity import ItemSimilarity
from recommendation.umatrix.matrix_model import MatrixBooleanPrefDataModel


with get_session() as session:
    sales = session.query(Sale).all()

samples = dict()
for sale in sales:
    if sale.user not in samples:
        samples[sale.user] = dict()
    samples[sale.user]['%s###%s' % (sale.brand, sale.product)] = 1.0

model = MatrixBooleanPrefDataModel(samples)
similarity = ItemSimilarity(model, loglikehood_coefficient)
recommender = ItemBasedRecommender(model, similarity)
items = recommender.recommend('13735329805', 3)

for item in items:
    print 'Recommend: ', item
    b_str = 'Because:'
    bs = recommender.recommended_because('13735329805', item, 2)
    for b in bs:
        b_str += ' ' + b
    print b_str

