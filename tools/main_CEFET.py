import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV, dump
from src.utils.results import df_results, save_pickle
from tqdm import tqdm

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../src/detectron2/projects/DensePose/'))

from src.utils.bayes_optim import bayes_search
from src.utils.feats import load_gei
from src.utils.plot_config import set_plot_config
from src.utils.results import df_results
from src.utils.search_spaces import (SVC_space, SVC_space_bayes, SVD_space, SVD_space_bayes,
                                     base_pipe, base_pipe_reduction)


def CEFET_fit(X,
              y,
              model,
              dim=(64, 48),
              crop_person=True,
              test_size=0.25,
              predict=False,
              random_state=None):

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    del X, y

    model.fit(X_train, y_train)

    del X_train, y_train

    y_pred = None
    score = None
    report = None
    cfn_mtx = None

    if predict:
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)

        y_pred = le.inverse_transform(y_pred)
        y_test = le.inverse_transform(y_test)

        report = classification_report(y_test, y_pred, labels=np.unique(y_test), zero_division=0)
        cfn_mtx = confusion_matrix(y_test, y_pred)

    return model, y_pred, score, report, cfn_mtx


def main(dim=(64, 48), crop_person=True, test_size=0.25, n_eval=64, n_trials=10, optim='random'):

    datapath = os.path.join(this_filepath, "../data/feats/database24_gei_480x640.pkl")
    X, y = load_gei(datapath, dim=dim, crop_person=crop_person)

    ### Define MODEL and params search space ###
    n_splits = 3  # for param search in bayesian optimization
    cv = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    pipe = base_pipe_reduction
    if optim.lower() == 'bayesian':
        red_space = SVD_space_bayes
        cls_space = SVC_space_bayes

        search_space = {**red_space, **cls_space}

        opt = BayesSearchCV(
            pipe,
            # (parameter space, # of evaluations)
            [
                (search_space, n_eval),
            ],
            scoring='accuracy',
            cv=cv,
            random_state=0)

    elif optim.lower() == 'random':
        red_space = SVD_space
        cls_space = SVC_space

        search_space = {**red_space, **cls_space}

        opt = RandomizedSearchCV(pipe,
                                 search_space,
                                 n_iter=n_eval,
                                 scoring='accuracy',
                                 cv=cv,
                                 n_jobs=-1,
                                 random_state=0)

    else:
        raise ValueError(f"`optim` must be `bayesian` or `random`: {optim}")

    model = dict()
    best_score = dict()
    elapsed_time = dict()
    report = dict()
    cfn_mtx = dict()

    for n_trial in tqdm(range(n_trials)):
        start_time = time.time()

        model[f'trial_{n_trial}'], _, best_score[f'trial_{n_trial}'], report[
            f'trial_{n_trial}'], cfn_mtx[f'trial_{n_trial}'] = CEFET_fit(X,
                                                                         y,
                                                                         opt,
                                                                         dim=dim,
                                                                         crop_person=crop_person,
                                                                         test_size=test_size,
                                                                         predict=True,
                                                                         random_state=n_trial)

        elapsed_time[f'trial_{n_trial}'] = time.time() - start_time

    return model, best_score, report, cfn_mtx, elapsed_time


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="MINDS main")

    parser.add_argument("--res_path",
                        type=str,
                        default=os.path.join(this_filepath, '..', 'results', 'CEFET'),
                        help="Directory to save results")

    parser.add_argument("--mod_path",
                        type=str,
                        default=os.path.join(this_filepath, '..', 'models', 'CEFET'),
                        help="Directory to save models")

    parser.add_argument("--n_eval",
                        type=int,
                        default=64,
                        help="number of evaluations for params optimization")

    parser.add_argument("--n_trials", type=int, default=10, help="number of runs of experiment")

    parser.add_argument("--test_size",
                        type=float,
                        default=0.25,
                        help="test size. Must be between 0 and 1. Default 0.25.")

    parser.add_argument("--optim",
                        type=str,
                        default='random',
                        help="Method to optimize hyperparams.")

    crop_person = True
    dim = (64, 48)

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
                                                          test_size=args.test_size,
                                                          n_eval=args.n_eval,
                                                          n_trials=args.n_trials,
                                                          optim=args.optim)

    for trial in opt.keys():
        df = df_results(opt[trial])
        df.to_csv(os.path.join(args.res_path, f"{trial}.csv"))

        # del opt[trial].specs['args']['func']

        f.write(f'\n\n{trial}: acc: {best_score[trial]} -- time elapsed: {elapsed_time[trial]}\n')
        f.write(f'{report[trial]}\n')
        f.write(f'Confusion mtx:\n{cfn_mtx[trial]}\n')

        if args.optim.lower() == 'bayesian':
            dump(opt[trial], os.path.join(args.mod_path, f'{trial}.gz'))
        else:
            joblib.dump(opt[trial], os.path.join(args.mod_path, f'{trial}.sav'))

    save_pickle(report, os.path.join(args.res_path, 'report.pkl'))
    save_pickle(cfn_mtx, os.path.join(args.res_path, 'cfn_mtx.pkl'))

    mean_score = np.array(list(best_score.values())).mean()
    std_score = np.array(list(best_score.values())).std()
    total_time = time.time() - start_time
    f.write(f'\nMean acc: {mean_score} +/- {std_score} -- elapsed time: {total_time} sec \n\n\n')
    f.close()
