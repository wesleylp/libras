import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from skopt import BayesSearchCV, dump
from skopt.space import Categorical, Integer, Real

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../src/detectron2/projects/DensePose/'))

from src.dataset.UFOP import UFOPDataset, gen_cv, labels
from src.utils.results import df_results, save_pickle

thisfile_dir = os.path.abspath(os.path.dirname(__file__))


def UFOP_fit(base_model,
             search_space,
             category,
             subset,
             dim=(64, 48),
             crop_person=True,
             predict=False):

    ufop_dir = os.path.join(this_filepath, '../data/LIBRAS-UFOP')

    ufop_dataset = UFOPDataset(ufop_dir, os.path.join(ufop_dir, 'labels.txt'))

    x_train, y_train = ufop_dataset.load_features(subset=subset,
                                                  mode='train',
                                                  categ=category,
                                                  shuffle=True,
                                                  dim=dim,
                                                  crop_person=crop_person,
                                                  flatten=True)

    x_valid, y_valid = ufop_dataset.load_features(subset=subset,
                                                  mode='val',
                                                  categ=category,
                                                  shuffle=True,
                                                  dim=dim,
                                                  crop_person=crop_person,
                                                  flatten=True)

    cv = gen_cv(y_train, y_valid)

    X = np.concatenate((x_train, x_valid))
    y = np.concatenate((y_train, y_valid))

    # to save memory
    del x_train, y_train, x_valid, y_valid

    le = LabelEncoder()
    y = le.fit_transform(y)

    opt = BayesSearchCV(
        base_model,
        # (parameter space, # of evaluations)
        search_space,
        cv=cv)

    opt.fit(X, y)

    y_pred = None
    score = None
    if predict:

        x_test, y_test = ufop_dataset.load_features(subset=subset,
                                                    mode='test',
                                                    categ=category,
                                                    shuffle=True,
                                                    dim=dim,
                                                    crop_person=crop_person,
                                                    flatten=True)
        y_test = le.transform(y_test)
        score = opt.score(x_test, y_test)

        y_pred = opt.predict(x_test)
        y_pred = le.inverse_transform(y_pred)
        y_test = le.inverse_transform(y_test)

        report = classification_report(y_test, y_pred, labels=np.unique(y_test))
        cfn_mtx = confusion_matrix(y_test, y_pred)

    return opt, y_pred, score, report, cfn_mtx


def main(category, dim, crop_person, n_eval=16):

    # pipeline class is used as estimator to enable
    # search over different model types
    base_pipe = Pipeline([('reduction', PCA()), ('model', SVC())])

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

    # (parameter space, # of evaluations)
    search_space = [
        (svc_SVD_space, n_eval),
    ]

    opt = dict()
    y_pred = dict()
    best_score = dict()
    time_elapsed = dict()
    report = dict()
    cfn_mtx = dict()

    subsets = ['set_1', 'set_2', 'set_3']
    for subset in subsets:
        start_time = time.time()
        opt[subset], y_pred[subset], best_score[subset], report[subset], cfn_mtx[subset] = UFOP_fit(
            base_pipe,
            search_space,
            category=category,
            subset=subset,
            dim=dim,
            crop_person=crop_person,
            predict=True)
        time_elapsed[subset] = time.time() - start_time

    return opt, best_score, report, cfn_mtx, time_elapsed


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="UFOP main")
    parser.add_argument("-c",
                        "--category",
                        type=str,
                        default='c2',
                        help="Category: [`c1`, `c2`, `c3`, `c4` or `all`]")
    parser.add_argument("--res_path",
                        type=str,
                        default=os.path.join(this_filepath, '..', 'results', 'UFOP'),
                        help="Directory to save results")

    parser.add_argument("--mod_path",
                        type=str,
                        default=os.path.join(this_filepath, '..', 'models', 'UFOP'),
                        help="Directory to save models")

    parser.add_argument("--n_eval",
                        type=int,
                        default=64,
                        help="number of evaluations for bayesian optimization")

    crop_person = True
    dim = (64, 48)

    args = parser.parse_args()
    os.makedirs(args.res_path, exist_ok=True)
    os.makedirs(args.mod_path, exist_ok=True)

    # results log
    f = open(os.path.join(args.res_path, f'results_{args.category}.txt'), "a")
    f.write('-' * 30)
    f.write('\nExperiment run on %s.\n' % datetime.now())
    f.write(f'args: {args}\n\n')

    start_time = time.time()
    opt, best_score, report, cfn_mtx, run_time = main(category=args.category,
                                                      dim=dim,
                                                      crop_person=crop_person,
                                                      n_eval=args.n_eval)

    for subset in opt.keys():
        df = df_results(opt[subset])
        df.to_csv(os.path.join(args.res_path, f"{subset}.csv"))

        # del opt[subset].specs['args']['func']

        f.write(f'\n\n{subset}: acc: {best_score[subset]} -- time elapsed: {run_time[subset]}\n')
        f.write(f'{report[subset]}\n')
        f.write(f'Confusion mtx:\n{cfn_mtx[subset]}\n')

        dump(opt[subset], os.path.join(args.mod_path, f'cat_{args.category}_{subset}.gz'))

    save_pickle(report, os.path.join(args.res_path, 'report.pkl'))
    save_pickle(cfn_mtx, os.path.join(args.res_path, 'cfn_mtx.pkl'))

    mean_score = np.array(list(best_score.values())).mean()
    std_score = np.array(list(best_score.values())).std()
    total_time = time.time() - start_time
    f.write(f'Mean acc: {mean_score} +/- {std_score} -- elapsed time: {total_time} sec \n\n\n')
    f.close()
