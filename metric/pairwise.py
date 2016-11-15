# coding=utf-8
import numpy as np
from scipy.spatial import distance


def euclidean_distances(x, y, squared=False, inverse=True):
    """
    Considering the rows of x (and y=x) as vectors, compute the
    distance matrix between each pair of vectors.

    An implementation of a "similarity" based on the Euclidean "distance"
    between two vectors x and y. Thinking of items as dimensions and
    preferences as points along those dimensions, a distance is computed using
    all items (dimensions) where both users have expressed a preference for
    that item. This is simply the square root of the sum of the squares of
    differences in position (preference) along each dimension.

    Parameters
    ----------
    x: array of shape (n_samples_1, n_features)

    y: array of shape (n_samples_2, n_features)

    squared: boolean, optional
        This routine will return squared Euclidean distances instead.

    inverse: boolean, optional
        This routine will return the inverse Euclidean distances instead.

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    #>>> x = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    #>>> # distance between rows of x
    #>>> euclidean_distances(x, x)
    #array([[ 1.        ,  0.29429806],
    #       [ 0.29429806,  1.        ]])
    #>>> # get distance to origin
    #>>> x = [[1.0, 0.0],[1.0,1.0]]
    #>>> euclidean_distances(x, [[0.0, 0.0]])
    #array([[ 0.5       ],
    #      [ 0.41421356]])
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as y, then you should just pre-compute the output and not even
    # call this function.

    if x is y:
        x = y = np.asanyarray(x)
    else:
        x = np.asanyarray(x)
        y = np.asanyarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Incompatible dimension for x and y matrices")

    if squared:
        return distance.cdist(x, y, 'sqeuclidean')

    xy = distance.cdist(x, y)
    return np.divide(1.0, (1.0 + xy)) if inverse else xy


euclidian_distances = euclidean_distances  # both spelling for backward compat


def pearson_correlation(x, y):
    """
    Considering the rows of x (and y=x) as vectors, compute the
    distance matrix between each pair of vectors.

    This correlation implementation is equivalent to the cosine similarity
    since the data it receives is assumed to be centered -- mean is 0. The
    correlation may be interpreted as the cosine of the angle between the two
    vectors defined by the users' preference values.

    Parameters
    ----------
    x: array of shape (n_samples_1, n_features)

    y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    #>>> x = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    #>>> # distance between rows of x
    #>>> pearson_correlation(x, x)
    #array([[ 1., 1.],
    #       [ 1., 1.]])
    #>>> pearson_correlation(x, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    #array([[ 0.39605902],
    #       [ 0.39605902]])
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as y, then you should just pre-compute the output and not even
    # call this function.

    if x is y:
        x = y = np.asanyarray(x)
    else:
        x = np.asanyarray(x)
        y = np.asanyarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Incompatible dimension for x and y matrices")

    xy = distance.cdist(x, y, 'correlation', 2)

    return 1 - xy


def cosine_distances(x, y):
    """
    Considering the rows of x (and y=x) as vectors, compute the
    distance matrix between each pair of vectors.

     An implementation of the cosine similarity. The result is the cosine of
     the angle formed between the two preference vectors.
     Note that this similarity does not "center" its data, shifts the user's
     preference values so that each of their means is 0. For this behavior,
     use Pearson Coefficient, which actually is mathematically
     equivalent for centered data.

    Parameters
    ----------
    x: array of shape (n_samples_1, n_features)

    y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    #>>> from scikits.crab.metrics.pairwise  import cosine_distances
    #>>> x = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    #>>> # distance between rows of x
    #>>> cosine_distances(x, x)
    #array([[ 1.,  1.],
    #      [ 1.,  1.]])
    #>>> cosine_distances(x, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    #array([[ 0.9606463],
    #       [ 0.9606463]])
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as y, then you should just pre-compute the output and not even
    # call this function.

    if x is y:
        x = y = np.asanyarray(x)
    else:
        x = np.asanyarray(x)
        y = np.asanyarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Incompatible dimension for x and y matrices")

    return 1.0 - distance.cdist(x, y, 'cosine')


def loglikehood_coefficient(n_items, x, y):
    """
    Considering the rows of x (and y=x) as vectors, compute the
    distance matrix between each pair of vectors.

    Parameters
    ----------
    n_items: int
        Number of items in the model.

    x: array of shape (n_samples_1, n_features)

    y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    #>>> x = [['a', 'b', 'c', 'd'],  ['e', 'f','g', 'h']]
    #>>> # distance between rows of x
    #>>> n_items = 7
    #>>> loglikehood_coefficient(n_items,x, x)
    #array([[ 1.,  0.],
    #      [ 0.,  1.]])
    #>>> n_items = 8
    #>>> loglikehood_coefficient(n_items, x, [['a', 'b', 'c', 'k']])
    #array([[ 0.67668852],
    #      [ 0.        ]])

    References
    ----------
    See http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.5962 and
    http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html.
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as y, then you should just pre-compute the output and not even
    # call this function.

    def safe_log(d):
        if d <= 0.0:
            return 0.0
        else:
            return np.log(d)

    def log_l(p, k, n):
        return k * safe_log(p) + (n - k) * safe_log(1.0 - p)

    def two_log_lambda(k1, k2, n1, n2):
        p = (k1 + k2) / (n1 + n2)
        return 2.0 * (log_l(k1 / n1, k1, n1) + log_l(k2 / n2, k2, n2)
                      - log_l(p, k1, n1) - log_l(p, k2, n2))

    if x is y:
        x = y = np.asanyarray(x)
    else:
        x = np.asanyarray(x)
        y = np.asanyarray(y)

    result = []

    i = 0
    for array_x in x:
        result.append([])
        for arrayY in y:
            xy = np.intersect1d(array_x, arrayY)

            if xy.size == 0:
                result[i].append(0.0)
            else:
                n_x = array_x.size
                n_y = arrayY.size
                if (n_x - xy.size == 0) or (n_items - n_y) == 0:
                    result[i].append(1.0)
                else:
                    log_likelihood = two_log_lambda(
                        float(xy.size),
                        float(n_x - xy.size),
                        float(n_y),
                        float(n_items - n_y)
                    )

                    result[i].append(1.0 - 1.0 / (1.0 + float(log_likelihood)))
        result[i] = np.asanyarray(result[i])
        i += 1

    return np.asanyarray(result)


if __name__ == '__main__':
    a = [['a', 'b', 'c', 'd'],  ['e', 'f', 'g', 'h']]
    r = loglikehood_coefficient(8, a, [['a', 'b', 'c', 'k']])
    print r