# coding=utf-8
"""
Strategies for users selection to be a
possible candidate to be member of a user neighborhood.

Please check the base.BaseUserNeighborhoodStrategy before
implement your own strategy.

"""
import numpy as np
from base import BaseUserCandidateStrategy
from recommendation.metric.pairwise import euclidean_distances
from recommendation.similarity.basic_similarity import UserSimilarity


class AllPossibleUserStrategy(BaseUserCandidateStrategy):
    """
    Returns
    --------
    Returns all users in the model.
    This strategy is not recommended for large datasets and
    it is the dummiest one.
    """
    def candidate_users(self, user_id, data_model, distance=None, nhood_size=None, **params):
        """
        Computes a neighborhood consisting of the  n users to a given user
        based on the strategy implemented in this method.

        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        data_model: DataModel instance
            The data model that will be the source for the possible
            candidates

        similarity: string
            The similarity to compute the neighborhood  (default = 'user_similarity')
            |user_similarity'|

        distance: function
            Pairwise metric to compute the similarity between the users.

        nhood_size: int
            The neighborhood size (default = None all users)
        """
        user_ids = data_model.user_ids()
        return user_ids[user_ids != user_id] if user_ids.size else user_ids


class NearestUserStrategy(BaseUserCandidateStrategy):
    """
    Returns
    --------
    Returns the neighborhood consisting of the nearest n
    users to a given user. "Nearest" in this context is
    defined by the Similarity.

    Parameters
    -----------
    user_id:  int or string
        ID of user for which to find most similar other users

    data_model: DataModel instance
        The data model that will be the source for the possible
        candidates

    distance: function
        Pairwise metric to compute the similarity between the users.

    nhood_size: int
        The neighborhood size (default = None all users)
    """
    def __init__(self):
        self.similarity = None

    def _set_similarity(self, data_model, distance, nhood_size):
        if not isinstance(self.similarity, UserSimilarity) or not distance == self.similarity.distance:
            nhood_size = nhood_size if not nhood_size else nhood_size + 1
            self.similarity = UserSimilarity(data_model, distance, nhood_size)

    def candidate_users(self, user_id, data_model, distance=None, nhood_size=None, **params):
        """
        Computes a neighborhood consisting of the  n users to a given
        user based on the strategy implemented in this method.
        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        data_model: DataModel instance
            The data model that will be the source for the possible
            candidates

        n_similarity: string
            The similarity to compute the neighborhood (Default = 'user_similarity')

        nhood_size: int
            The neighborhood size (default = None all users)

        Optional Parameters
        --------------------
        minimal_similarity: float
            minimal similarity required for neighbors (default = 0.0)

        sampling_rate: int
            percentage of users to consider when building neighborhood
                (default = 1)
        """
        minimal_similarity = params.get('minimal_similarity', 0.0)

        # set the nhood_size at Similarity , and use Similarity to get the top_users
        if distance is None:
            distance = euclidean_distances
        self._set_similarity(data_model, distance, nhood_size)

        neighborhood = [to_user_id for to_user_id, score in self.similarity[user_id]
                        if not np.isnan(score) and score >= minimal_similarity and user_id != to_user_id]

        return neighborhood
