from copy import deepcopy as cp

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC


DEFAULT_CONFIG = {

    'vectorizer': {
        'ngram_range': (3, 6),
        'max_df': 0.2,
        'min_df': 5,
        'norm': 'l2',
        'analyzer': 'char'
    },

    'feature_selector': {
        'criterion': 'entropy',
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 3
    },

    'feature_selection': {
        'max_features': 10
    },

    'classifier': {
        'model': None
    }

}


class Model:

    def __init__(self, config=DEFAULT_CONFIG):
        self.__dict__.update(config)
        self.__do_select = True if self.__dict__['feature_selector'] \
                           and self.__dict__['feature_selection'] \
                           else False
        #self.selector = DecisionTreeClassifier(**config['feature_selector'])
        self.selector = LogisticRegression(**config['feature_selector'])
        self.select = None

        self.vec = TfidfVectorizer(**config['vectorizer'])

        model_params = \
            {key: val for key, val in config['classifier'].items() if key != 'model'}

        self.model = self.__dict__['classifier']['model'](**model_params)


    def train(self, X, Y):

        if not self.model:
            raise ValueError('A model must be defined as `config["classifier"]["model"]`')

        X_vec = self.vec.fit_transform(X, Y)

        if self.__do_select:
            self.selector.fit(X_vec, Y)
            self.select = SelectFromModel(
                self.selector,
                prefit=True,
                max_features=self.__dict__['feature_selection']['max_features']
            )
            X_select = self.select.transform(X_vec)
        else:
            X_select = X_vec

        self.model.fit(X_select, Y)


    def predict(self, X):
        if not self.model:
            raise ValueError('A model must be defined as `config["classifier"]["model"]`')
        elif self.__do_select and not self.select:
            raise ValueError('The feature selector did not initialize correctly.')
        X_vec = self.vec.transform(X)
        X_select = self.select.transform(X_vec) if self.__do_select \
                   else X_vec
        return self.model.predict(X_select)


if __name__ == '__main__':

    config = {
        'vectorizer': {
            'ngram_range': (1, 1),
            'max_df': 1.0,
            'min_df': 1,
            'norm': 'l2',
            'analyzer': 'word'
        },
        'feature_selector': {
            'criterion': 'entropy',
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'feature_selection': {
            'max_features': 1
        },
        'classifier': {
            'dual': True,
            'model': LinearSVC
        }
    }

    target_accuracy = 0.85

    model = Model(config=config)


    import random

    from sklearn.metrics import precision_score, recall_score


    def new_record(category):

        if category == 1:
            text = ['a' + str(random.randrange(1, 101)) for _ in range(50)] \
                   + ['a' + str(1 if random.random() < target_accuracy else 0)
                      for _ in range(10)]
        else:
            text = ['a' + str(random.randrange(1, 101)) for _ in range(50)] \
                   + ['a' + str(0 if random.random() < target_accuracy else 1)
                      for _ in range(10)]

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

