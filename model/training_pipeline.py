import os
import random
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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



vectorizers = {

    'unigram_vectorizer': TfidfVectorizer(
        ngram_range = (1, 1),
        max_df = 1.0,
        min_df = 1,
        norm = 'l2',
        analyzer = 'word'
    ),

    'unigram_vectorizer_noidf': TfidfVectorizer(
        ngram_range = (1, 1),
        max_df = 1.0,
        min_df = 1,
        norm = 'l2',
        analyzer = 'word',
        use_idf = False
    ),

    'char_vectorizer': TfidfVectorizer(
        ngram_range = (3, 5),
        max_df = 0.3,
        min_df = 5,
        norm = 'l2',
        analyzer = 'char'
    ),

    'char_vectorizer_noidf': TfidfVectorizer(
        ngram_range = (3, 5),
        max_df = 1.0,
        min_df = 1,
        norm = 'l2',
        analyzer = 'char'
    ),

}




selectors = {

    'selector_lr': LogisticRegression(
        penalty = 'l2',
        max_iter = 300,
        solver = 'sag',
        n_jobs = 4,
        multi_class = 'ovr'
    ),

    'selector_svm': LinearSVC(dual = False)

}

reducer_params = {
    'reducer_2000': {'prefit': False, 'max_features': 2000},
    'reducer_3000': {'prefit': False, 'max_features': 3000},
    'reducer_5000': {'prefit': False, 'max_features': 5000},
}

classifiers = {

    'classifier_lr': LogisticRegression(
        penalty = 'l1',
        max_iter = 100,
        solver = 'saga',
        n_jobs = 4,
        multi_class = 'multinomial'
    ),

    'classifier_svm': LinearSVC(dual = False),

    'classifier_nb': MultinomialNB()

}



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

    components = dict([])
    full_params = dict([])
    run_name__parts = []
    for abstract_name, component_name, registry in [
        ('vectorizer', 'char_vectorizer_noidf', vectorizers),
        ('reducer', 'reducer_2000', reducer_params),
        ('selector', 'selector_svm', selectors),
        #('classifier', 'classifier_nb', classifiers),
        ('classifier', 'classifier_svm', classifiers),
    ]:
        full_params[f'{abstract_name}'] = component_name
        run_name__parts.append(component_name)
        record = registry[component_name]

        if not isinstance(record, dict):
            full_params.update({
                f'{abstract_name}__{key}': val
                for key, val in record.get_params().items()
            })
        else:
            full_params.update({
                f'{abstract_name}__{key}': val
                for key, val in record.items()
            })
        components[abstract_name] = registry[component_name]
    run_name = '.'.join(run_name__parts)


    kf = KFold(n_splits=5, shuffle=True)
    for _n_fold, (train_idx, test_idx) in enumerate(kf.split(X)):

        try:
            n_fold = _n_fold + 1
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
            ('vectorizer', components['vectorizer']),
            ('reducer', SelectFromModel(
                components['selector'],
                **components['reducer']
            )),
            ('classifier', components['classifier'])
        ])

        pipe.fit(X_train, Y_train)
        score = pipe.score(X_test, Y_test)

        print([Y_test.count(y) for y in set(Y_test)], score)

        mlflow.log_metric('accuracy', score)
        mlflow.end_run()

        mlflow.sklearn.save_model(pipe, f"artifacts/{run_name}/{n_fold}")


