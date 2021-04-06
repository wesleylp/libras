import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from skopt import BayesSearchCV, dump, load
from skopt.space import Categorical, Integer, Real

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../src/detectron2/projects/DensePose/'))

from src.dataset.UFOP import UFOPDataset, gen_cv, labels
from src.utils.results import df_results

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

        y_pred = opt.predict(x_test)
        y_pred = le.inverse_transform(y_pred)

        score = opt.score(x_test, y_test)

    return opt, y_pred, score


def main(category, dim, crop_person):

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
        (svc_SVD_space, 1024),
    ]

    opt = dict()
    y_pred = dict()
    score = dict()
    subsets = ['set_1', 'set_2', 'set_3']
    for subset in subsets:
        opt[subset], y_pred[subset], score[subset] = UFOP_fit(base_pipe,
                                                              search_space,
                                                              category=category,
                                                              subset=subset,
                                                              dim=dim,
                                                              crop_person=crop_person,
                                                              predict=True)

    mean_score = np.array(list(score.values())).mean()
    std_score = np.array(list(score.values())).std()

    return opt, mean_score, std_score


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
                        default=os.path.join(this_filepath, 'results', 'UFOP'),
                        help="Directory to save results")

    parser.add_argument("--mod_path",
                        type=str,
                        default=os.path.join(this_filepath, 'models', 'UFOP'),
                        help="Directory to save models")

    crop_person = True
    dim = (64, 48)

    args = parser.parse_args()

    opt, mean_score, std_score = main(category=args.category, dim=dim, crop_person=crop_person)

    os.makedirs(args.res_path, exist_ok=True)

    for subset in opt.keys():
        df = df_results(opt[subset])
        df.to_csv(os.path.join(args.res_path, f"{subset}.csv"))

        # del opt[subset].specs['args']['func']

        dump(opt[subset], os.path.join(args.mod_path, f'{subset}.gz'))

    print(f"Results for: Category {args.category}")
    print(f"Mean acc: {mean_score} +/- {std_score}")
    print('end')
