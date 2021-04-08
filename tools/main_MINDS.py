import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from skopt import BayesSearchCV, dump
from skopt.space import Categorical, Integer, Real

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../src/detectron2/projects/DensePose/'))

from src.dataset.MINDS import MINDSDataset
from src.utils.results import df_results, save_pickle

thisfile_dir = os.path.abspath(os.path.dirname(__file__))


def train_minds(X, y, model, test_size=0.25, random_state=None, predict=False):

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = None
    score = None
    if predict:
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)

        y_pred = le.inverse_transform(y_pred)
        y_test = le.inverse_transform(y_test)

        report = classification_report(y_test, y_pred, labels=np.unique(y_test))
        cfn_mtx = confusion_matrix(y_test, y_pred)

    return model, y_pred, score, report, cfn_mtx


def main(dim=(64, 48),
         crop_person=True,
         exclude={
             3: 'all',
             4: ["Filho"],
             9: 'all'
         },
         n_eval=16,
         n_trials=10):
    minds_dataset = MINDSDataset(os.path.join(this_filepath, "../data/MINDS-Libras_RGB-D/"))

    X, y = minds_dataset.load_features(
        exclude=exclude,
        dim=dim,
        crop_person=crop_person,
        shuffle=True,
        flatten=True,
    )

    # pipeline class is used as estimator to enable
    # search over different model types
    base_pipe = Pipeline([('reduction', TruncatedSVD()), ('model', SVC())])

    svc_SVD_space = {
        'reduction': Categorical([
            TruncatedSVD(random_state=0),
        ]),
        'reduction__n_components': Integer(2, 150),
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    ### Define MODEL and params search space ###
    n_splits = 3  # for param search in bayesian optimization
    cv = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    opt = BayesSearchCV(
        base_pipe,
        # (parameter space, # of evaluations)
        [
            (svc_SVD_space, n_eval),
        ],
        scoring='accuracy',
        cv=cv)

    model = dict()
    best_score = dict()
    elapsed_time = dict()
    report = dict()
    cfn_mtx = dict()

    for n_trial in range(n_trials):
        start_time = time.time()
        model[f'trial_{n_trial}'], _, best_score[f'trial_{n_trial}'], report[
            f'trial_{n_trial}'], cfn_mtx[f'trial_{n_trial}'] = train_minds(X.copy(),
                                                                           y.copy(),
                                                                           opt,
                                                                           test_size=0.25,
                                                                           random_state=n_trial,
                                                                           predict=True)

        elapsed_time[f'trial_{n_trial}'] = time.time() - start_time

    return model, best_score, report, cfn_mtx, elapsed_time


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="MINDS main")

    parser.add_argument("--res_path",
                        type=str,
                        default=os.path.join(this_filepath, '..', 'results', 'MINDS'),
                        help="Directory to save results")

    parser.add_argument("--mod_path",
                        type=str,
                        default=os.path.join(this_filepath, '..', 'models', 'MINDS'),
                        help="Directory to save models")

    parser.add_argument("--n_eval",
                        type=int,
                        default=64,
                        help="number of evaluations for bayesian optimization")

    parser.add_argument("--n_trials", type=int, default=10, help="number of runs ofexperiment")

    crop_person = True
    dim = (64, 48)
    exclude = {3: 'all', 4: ["Filho"], 9: 'all'}

    args = parser.parse_args()

    os.makedirs(args.res_path, exist_ok=True)
    os.makedirs(args.mod_path, exist_ok=True)

    # results log
    f = open(os.path.join(args.res_path, 'results.txt'), "a")
    f.write('-' * 30)
    f.write('\nExperiment run on %s.\n' % datetime.now())
    f.write(f'args: {args}\n\n')

    start_time = time.time()
    opt, best_score, report, cfn_mtx, elapsed_time = main(dim=dim,
                                                          crop_person=crop_person,
                                                          exclude=exclude,
                                                          n_eval=args.n_eval,
                                                          n_trials=args.n_trials)

    for trial in opt.keys():
        df = df_results(opt[trial])
        df.to_csv(os.path.join(args.res_path, f"{trial}.csv"))

        # del opt[trial].specs['args']['func']

        f.write(f'\n\n{trial}: acc: {best_score[trial]} -- time elapsed: {elapsed_time[trial]}\n')
        f.write(f'{report[trial]}\n')
        f.write(f'Confusion mtx:\n{cfn_mtx[trial]}\n')

        dump(opt[trial], os.path.join(args.mod_path, f'{trial}.gz'))

    save_pickle(report, os.path.join(args.res_path, 'report.pkl'))
    save_pickle(cfn_mtx, os.path.join(args.res_path, 'cfn_mtx.pkl'))

    mean_score = np.array(list(best_score.values())).mean()
    std_score = np.array(list(best_score.values())).std()
    total_time = time.time() - start_time
    f.write(f'\nMean acc: {mean_score} +/- {std_score} -- elapsed time: {total_time} sec \n\n\n')
    f.close()
