from copy import deepcopy as cp

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC


class Model:

    def __init__(
        self,
        ngram_range=(3, 6),
        max_df=0.2,
        min_df=5,
        norm='l2',
        analyzer='char',

        criterion='entropy',
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3

    ):

        self.selector = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.select = None

        self.vec = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            norm=norm,
            analyzer=analyzer
        )

        self.model = LinearSVC(dual=True)


    def train(self, X, Y):
        X_vec = self.vec.fit_transform(X, Y)
        self.selector.fit(X_vec, Y)
        self.select = SelectFromModel(self.selector, prefit=True)
        X_select = self.select.transform(X_vec)
        self.model.fit(X_select, Y)


    def predict(self, X):
        X_vec = self.vec.transform(X)
        X_select = self.select.transform(X_vec)
        return self.model.predict(X_select)


if __name__ == '__main__':

    model = Model(
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        norm='l2',
        analyzer='word',

        criterion='entropy',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )

    import random

    from sklearn.metrics import precision_score, recall_score

    def new_record(category):
        if category == 1:
            text = ['a' + str(random.randrange(1, 101)) for _ in range(50)] \
                   + ['a' + str(random.randrange(5, 10)) for _ in range(30)] \
                   + ['a' + str(1) for _ in range(10)]
        else:
            text = ['a' + str(random.randrange(1, 101)) for _ in range(50)] \
                   + ['a' + str(random.randrange(0, 5)) for _ in range(30)] \
                   + ['a' + str(0) for _ in range(10)]
        return ' '.join(text)


    X1 = [new_record(0) for _ in range(5000)]
    X2 = [new_record(1) for _ in range(5000)]
    X = X1 + X2
    Y = [0 for _ in range(5000)] + [1 for _ in range(5000)]

    kf = KFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        #print(len(train_idx), len(test_idx))
        X_train = [X[idx] for idx in train_idx]
        X_test = [X[idx] for idx in test_idx]
        Y_train = [Y[idx] for idx in train_idx]
        Y_test = [Y[idx] for idx in test_idx]

        model.train(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print([Y_test.count(y) for y in set(Y_test)], precision_score(Y_test, Y_pred), recall_score(Y_test, Y_pred))

