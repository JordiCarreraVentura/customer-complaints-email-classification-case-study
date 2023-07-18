import random
from collections import Counter, defaultdict
from doctest import testmod

from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids


class DatasetBalancer:

    def __init__(self):
        return


    def __make_disambiguation(self, X, Y, margin=0.3):
        preferences = defaultdict(Counter)
        for x, y in zip(X, Y):
            key = x if isinstance(x, str) else tuple(x)
            preferences[key][y] += 1

        disambiguations = dict([])
        for x, ys in preferences.items():
            y_dist = ys.most_common()
            best_y, best_freq = y_dist[0]
            if len(y_dist) == 1:
                disambiguations[x] = best_y
                continue

            next_freq, next_y = y_dist[1]
            if next_freq / best_freq <= margin:
                disambiguations[x] = best_y
            else:
                disambiguations[x] = None

        return disambiguations


    def transform(self, X, Y):
        """
        Parameters
        ----------

        Subsamples from X and Y in parallel (if the i-th instance of X is fil-
        tered out, then so is Y's i-th instance) so that the following criteria
        are met:
        1. All majority classes in Y are subsampled down to the number of ins-
           tances of the minority class.
        2. For instances `x \in X`  associated with multiple `y \in Y`, keep
           only the most frequent `y` if it's significantly more frequently
           than the rest; discard the sample otherwise.
        3. Stratified sampling for all Y based on the clusters in their asso-
           ciated `Y`. Elbow method for selecting the number of clusters in
           each Y.
        4. Duplicate removal within each class Y, implemented as diversity sam-
           pling within each cluster: for each class, an instance is collected
           for each cluster in each iteration. During that one iteration, the
           order of candidates for sampling in that cluster is weighted by di-
           versity: from the cluster elements, the one *least similar* to al-
           ready sampled ones will be sampled next.

        :param X: ....
        :type  X: list[str]

        :param X: ....
        :type  X: list[str]

        :return:  Stratified and diversity-sampled subsets of X and Y.
        :rtype :  tuple[list[str], list[str]]


        Tests
        -----

        1. Balanced dataset (nothing to do, passthrough)

        >>> db = DatasetBalancer()
        >>> XY = [
        ...   ([y for _ in range(5)], y)
        ...   for y in range(3)
        ...   for _ in range(3)
        ... ]
        >>> X_, Y_ = list(zip(*XY))
        >>> X, Y = db.transform(X_, Y_)
        >>> assert set(Y) == set(Y_)
        >>> assert [Y_.count(y) == Y.count(y) for y in set(Y)]


        2. Unbalanced dataset

        >>> db = DatasetBalancer()
        >>> XY = [
        ...   ([y for _ in range(5)], y)
        ...   for y in range(3)
        ...   for _ in range(y + 1)
        ... ]
        >>> X_, Y_ = list(zip(*XY))
        >>> X, Y = db.transform(X_, Y_)
        >>> assert set(Y) == set(Y_)
        >>> assert len(set([Y.count(y) for y in set(Y)])) == 1


        3. Deduplication

        >>> db = DatasetBalancer()
        >>> XY = [
        ...   ([y for _ in range(5)], y)
        ...   for y in range(3)
        ...   for _ in range(3)
        ... ]
        >>> XY += [([1 for _ in range(5)], 2) for _ in range(2)]
        >>> XY += [([3 for _ in range(5)], 2) for _ in range(2)]
        >>> XY += [([3 for _ in range(5)], 1) for _ in range(1)]
        >>> X_, Y_ = list(zip(*XY))
        >>> X, Y = db.transform(X_, Y_)
        >>> assert 1 not in set(Y)
        >>> assert [Y_.count(y) == Y.count(y) for y in set(Y)]


        """
        Y_dist = Counter(Y)
        XY = list(zip(X, Y))
        disambiguations = self.__make_disambiguation(X, Y)
        random.seed(181)
        random.shuffle(XY)
        n_samples = min(Y_dist.values())
        Y_sampled = Counter()
        XY_ = []
        #print(disambiguations)
        for x, y in XY:
            #print(Y_sampled)
            if Y_sampled[y] < n_samples \
            and disambiguations[x if isinstance(x, str) else tuple(x)] == y:
                XY_.append((x, y))
                Y_sampled[y] += 1
            if sum(Y_sampled.values()) / len(Y_dist) >= n_samples:
                #print(n_samples, sum(Y_sampled.values()), len(Y_sampled), sum(Y_sampled.values()) / len(Y_sampled))
                break
        X_, Y_ = list(zip(*XY_))
        return X_, Y_




# class Resampler:
#
#     def __init__(self):
#         return
#
#     def transform(self, X, Y):
#         print('Original dataset shape %s' % Counter(Y))
#
#         cc = ClusterCentroids(
#             estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
#         )
#
#         X_res, Y_res = cc.fit_resample(X, Y)
#
#         print('Resampled dataset shape %s' % Counter(Y_res))
#
#         return X_res, Y_res
#
#


if __name__ == '__main__':

    testmod()


#     db = DatasetBalancer()
#
#     # Balanced
#     XY = [
#         ([y for _ in range(5)], y)
#         for y in range(3)
#         for _ in range(3)
#     ]
#     X_, Y_ = list(zip(*XY))
#     X, Y = db.transform(X_, Y_)
#     for x, y in zip(X, Y):
#         print(x, y)
# #     assert set(Y) == set(Y_)
# #     assert [Y_.count(y) == Y.count(y) for y in set(Y)]
#
#
#
#     # Unbalanced
#     XY = [
#         ([y for _ in range(5)], y)
#         for y in range(3)
#         for _ in range(y + 1)
#     ]
#     X_, Y_ = list(zip(*XY))
#     X, Y = db.transform(X_, Y_)
#     for x, y in zip(X, Y):
#         print(x, y)
#     print(set([Y_.count(y) for y in set(Y)]))
#     print(len(set([Y_.count(y) for y in set(Y)])))



    # Duplicate X

    db = DatasetBalancer()

    # Balanced
    XY = [
        ([y for _ in range(5)], y)
        for y in range(3)
        for _ in range(3)
    ]
    XY += [([1 for _ in range(5)], 2) for _ in range(2)]
    XY += [([3 for _ in range(5)], 2) for _ in range(2)]
    XY += [([3 for _ in range(5)], 1) for _ in range(1)]
#     for x in XY:
#         print(x)


    X_, Y_ = list(zip(*XY))
    X, Y = db.transform(X_, Y_)
#     for x, y in zip(X, Y):
#         print(x, y)
    assert 1 not in set(Y)
    assert [Y_.count(y) == Y.count(y) for y in set(Y)]






    """
    rsmplr = Resampler()

    # Balanced
    XY = [
        ([y for _ in range(5)], y)
        for y in range(3)
        for _ in range(3)
    ]
    X_, Y_ = list(zip(*XY))
    X, Y = rsmplr.transform(X_, Y_)
    X = X.tolist()
    Y = Y.tolist()
    assert set(Y) == set(Y_)
    assert [Y_.count(y) == Y.count(y) for y in set(Y)]



    # Unbalanced
    XY = [
        ([y for _ in range(5)], y)
        for y in range(3)
        for _ in range(y + 1)
    ]
    X_, Y_ = list(zip(*XY))
    X, Y = rsmplr.transform(X_, Y_)
    X = X.tolist()
    Y = Y.tolist()
    for x, y in zip(X, Y):
        print(x, y)
    print(set([Y_.count(y) for y in set(Y)]))
    print(len(set([Y_.count(y) for y in set(Y)])))
    """




