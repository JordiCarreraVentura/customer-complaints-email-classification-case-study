import os
import random
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


HOME = 'customer-complaints-email-classification-case-study'
curr = os.path.dirname(__file__)
while True:
    if curr.endswith(HOME):
        break
    curr = os.path.dirname(curr)
sys.path.append(curr)

from api.mlflow_interfaces import MlflowTrackingServer
from utils import get_credential



if __name__ == '__main__':

    dataset = pd.read_csv('data/dataset.dedup.resampled.csv')
    X, Y = dataset['narrative'], dataset['product']

    XY = list(zip(X, Y))
    random.shuffle(XY)
    XY = XY[:5000]
    X, Y = list(zip(*XY))

    host_file = os.path.join(curr, 'doc', 'mlflow.tracking_server.host.txt')
    tracking_server = MlflowTrackingServer(host=get_credential(host_file))
    tracking_server.createif_experiment(HOME)


    unigram_vectorizer__params = {
        'ngram_range': (1, 1),
        'max_df': 1.0,
        'min_df': 1,
        'norm': 'l2',
        'analyzer': 'word'
    }
    char_vectorizer__params = {
        'ngram_range': (3, 5),
        'max_df': 0.3,
        'min_df': 5,
        'norm': 'l2',
        'analyzer': 'char'
    }
#     selection_lr__params = {
#         'penalty': 'l2',
#         'max_iter': 300,
#         'solver': 'sag',
#         'n_jobs': 4,
#         'multi_class': 'ovr',
#     }
    selection_svm__params = {
        'dual': False
    }
    lowdim_reducer__params = {
        'prefit': False,
        'max_features': 2000
    }
#     classifier_params = {
#         'penalty': 'l1',
#         'max_iter': 100,
#         'solver': 'saga',
#         'n_jobs': 4,
#         'multi_class': 'multinomial'
#     }
    classifier_params = {
        'dual': False
    }

    unigram_vectorizer = TfidfVectorizer(**unigram_vectorizer__params)
    char_vectorizer = TfidfVectorizer(**char_vectorizer__params)
    #selector = LogisticRegression(**selection_lr__params)
    selector = LinearSVC(**selection_svm__params)
    lowdim_reducer = SelectFromModel(selector, **lowdim_reducer__params)
    #classifier = LogisticRegression(**classifier_params)
    classifier = LinearSVC(**classifier_params)


    vectorizer_params = char_vectorizer__params
    selector_params = selection_svm__params
    reducer_params = lowdim_reducer__params
    classifier_params = classifier_params
    vectorizer = char_vectorizer
    reducer = lowdim_reducer
    classifier = classifier

    full_params = dict([])
    for component_name, params in [
        ('vectorizer', vectorizer_params),
        ('selector', selector_params),
        ('reducer', reducer_params),
        ('classifier', classifier_params),
    ]:
        full_params.update({
            f'{component_name}__{key}': val
            for key, val in params.items()
        })

    kf = KFold(n_splits=5, shuffle=True)
    for _n_fold, (train_idx, test_idx) in enumerate(kf.split(X)):

        try:
            n_fold = _n_fold + 1
            run_name = 'char-svm-svm'
            mlflow.start_run(run_name=f'{run_name}/fold={n_fold}')
        except Exception:
            pass

        mlflow.log_params(full_params)

        #print(len(train_idx), len(test_idx))
        X_train = [X[idx] for idx in train_idx]
        X_test = [X[idx] for idx in test_idx]
        Y_train = [Y[idx] for idx in train_idx]
        Y_test = [Y[idx] for idx in test_idx]

        pipe = Pipeline([
            #('vectorizer', unigram_vectorizer),
            ('vectorizer', vectorizer),
            ('reducer', reducer),
            ('classifier', classifier)
        ])

        pipe.fit(X_train, Y_train)
        score = pipe.score(X_test, Y_test)

        print([Y_test.count(y) for y in set(Y_test)], score)

        mlflow.log_metric('accuracy', score)
        mlflow.end_run()



"""
    # LR3: 81.5% micro-averaged precision and recall, lowest variance

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

"""

