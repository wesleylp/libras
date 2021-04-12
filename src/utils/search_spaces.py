import numpy as np
from scipy.stats import loguniform
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skopt.space import Categorical, Integer, Real

# pipeline class is used as estimator to enable
# search over different model types
base_pipe = Pipeline([('model', SVC())])
base_pipe_reduction = Pipeline([('reduction', TruncatedSVD()), ('model', SVC())])

SVD_space_bayes = {
    'reduction': Categorical([
        TruncatedSVD(random_state=0),
    ]),
    'reduction__n_components': Integer(2, 150),
}

SVC_space_bayes = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1, 8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf'])
}

SVD_space = {
    'reduction': [
        TruncatedSVD(random_state=0),
    ],
    'reduction__n_components': np.arange(2, 150, dtype=int),
}

SVC_space = {
    'model': [
        SVC(random_state=0),
    ],
    'model__C': loguniform(
        1e-6,
        1e+6,
    ),
    'model__gamma': loguniform(1e-6, 1e+1),
    'model__degree': np.arange(1, 8, dtype=int),
    'model__kernel': ['linear', 'poly', 'rbf'],
}

KNN_space = {
    'model': [KNeighborsClassifier()],
    'model__n_neighbors': np.arange(1, 6, dtype=int),
}

RF_space = {
    'model': [
        RandomForestClassifier(max_depth=None, random_state=0, criterion='gini'),
    ],
    'model__n_estimators': np.arange(250, 400, dtype=int),
}
