import os
import random
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from Model import Model




if __name__ == '__main__':

    dataset = pd.read_csv('data/dataset.dedup.resampled.csv')
    X, Y = dataset['narrative'], dataset['product']

    XY = list(zip(X, Y))
    random.shuffle(XY)
    XY = XY[:5000]
    X, Y = list(zip(*XY))

    config = {
        'vectorizer': {
            'ngram_range': (3, 5),
            'max_df': 0.33,
            'min_df': 20,
            'norm': 'l2',
            'analyzer': 'char'
        },
        'feature_selector': {
            'criterion': 'entropy',
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'feature_selection': {
            'max_features': 50000
        },
        'classifier': {
            'dual': True,
            'model': LinearSVC
        }
    }

    config['vectorizer']: {
        'ngram_range': (1, 1),
        'max_df': 1.0,
        'min_df': 1,
        'norm': 'l2',
        'analyzer': 'word'
    }
    config['feature_selector'] = {
        'penalty': 'l2',
        'max_iter': 300,
        'solver': 'sag',
        'n_jobs': 4,
        'multi_class': 'ovr',
    }
    config['feature_selection'] = {
        'max_features': 3000
    }
    config['classifier']: {
        'penalty': 'l1',                # slightly better than L2
        #'penalty': 'l2',
        'max_iter': 100,
        'solver': 'saga',
        'n_jobs': 4,
        #'multi_class': 'ovr',
        'multi_class': 'multinomial',   # better than 'ovr'
        'model': LogisticRegression,
    }

    # max_features = 1000 + char vectorizer = ~78%
    # max_features = 1000 + word vectorizer = ~75%
    # max_features = 1000 + word vectorizer + (max_df = 0.2) + (min_df = 3) = ~76%

    model = Model(config=config)

    kf = KFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        #print(len(train_idx), len(test_idx))
        X_train = [X[idx] for idx in train_idx]
        X_test = [X[idx] for idx in test_idx]
        Y_train = [Y[idx] for idx in train_idx]
        Y_test = [Y[idx] for idx in test_idx]

        model.train(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print(
            [Y_test.count(y) for y in set(Y_test)],
            precision_score(Y_test, Y_pred, average='micro'),
            recall_score(Y_test, Y_pred, average='micro')
        )




"""

    # LR1: 81% micro-averaged precision and recall

    config['vectorizer']: {
        'ngram_range': (1, 1),
        'max_df': 0.2,
        'min_df': 3,
        'norm': 'l2',
        'analyzer': 'word'
    }
    config['feature_selector'] = dict([])
    config['feature_selection'] = dict([])
    config['classifier']: {
        'penalty': 'l1',                # slightly better than L2
        #'penalty': 'l2',
        'max_iter': 100,
        'solver': 'saga',
        'n_jobs': 4,
        #'multi_class': 'ovr',
        'multi_class': 'multinomial',   # better than 'ovr'
        'model': LogisticRegression,
    }
"""



"""

    # LR2: 82% micro-averaged precision and recall

    config['vectorizer']: {
        'ngram_range': (1, 1),
        'max_df': 1.0,
        'min_df': 1,
        'norm': 'l2',
        'analyzer': 'word'
    },

    ... como LR1 ...

"""