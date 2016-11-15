# coding=utf-8
u"""
User: xulin
Date: 13-11-21
"""
from numpy import matrix, zeros
from recommendation.model.sale import Sale
from recommendation.model.session import get_session


class UtilityMatrix(object):
    def __init__(self):
        self.samples = list()
        self.features = list()
        self.matrix = None

    def load_data_from_db(self):
        with get_session() as session:
            sales = session.query(Sale).all()

            for sale in sales:
                feature = UtilityMatrix.gen_bp_key(sale.brand, sale.product)
                if feature not in self.features:
                    self.features.append(feature)
                if sale.user not in self.samples:
                    self.samples.append(sale.user)

            self.matrix = matrix(zeros((len(self.samples), len(self.features))))

            for sale in sales:
                row = self.samples.index(sale.user)
                col = self.features.index(UtilityMatrix.gen_bp_key(sale.brand, sale.product))
                self.matrix[row, col] = 1

    @staticmethod
    def gen_bp_key(brand, product):
        return '%s###%s' % (brand, product)


if __name__ == '__main__':
    um = UtilityMatrix()
    um.load_data_from_db()
    print um.matrix
    print um.samples[0]
    print um.features[0]
    print 'samples len ', len(um.samples)

    i = um.samples.index('15958538990')
    print um.matrix[i, :]
    for elem in um.features[2:8]:
        print elem