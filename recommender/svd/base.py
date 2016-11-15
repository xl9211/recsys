# coding=utf-8
"""
Generalized Recommender models amd utility classes.

This module contains basic memory recommender interfaces used throughout
the whole scikit-crab package as also utility classes.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be sub-classed).
"""
from recommendation.recommender.base import BaseRecommender


#===========================
#Matrix Factorization Recommender Interface
class SVDRecommender(BaseRecommender):
    def factorize(self):
        """
        Factorize the ratings matrix with a factorization
         technique implemented in this method.

        Parameters
        -----------

        Returns
        -----------
        """
        raise NotImplementedError("ItemRecommender is an abstract class.")

    def train(self):
        """
        Train the recommender with the matrix factorization method chosen.

        Parameters
        -----------

        Returns
        ----------

        """
        raise NotImplementedError("ItemRecommender is an abstract class.")
