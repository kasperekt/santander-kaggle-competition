import pickle
import numpy as np

from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from model.catboost import get_model


def get_params_filepath(Explorer):
    return '../out/params.{}.pickle'.format(Explorer.name)


def cross_val(X, y, params, n_splits=10, verbose=False):
    skf = StratifiedKFold(n_splits, shuffle=True)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        model = get_model(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=verbose)
        score = model.score(X_val, y_val)
        scores.append(score)

    return sum(scores)/n_splits


def find_params(X, y, Explorer, eval_set=None):
    param_grid = ParameterGrid({'learning_rate': np.logspace(-2, -1, 4) * 3,
                                'iterations': [1],
                                'l2_leaf_reg': np.logspace(-19, -18, 3),
                                'depth': [2, 3]})

    params_file = get_params_filepath(Explorer)
    explorer = Explorer(param_grid)
    params = explorer.fit(X, y, eval_set=eval_set, params_file=params_file)
    return params


def load_params(Explorer):
    params_file = get_params_filepath(Explorer)

    with open(params_file, 'rb') as params_fp:
        params = pickle.load(params_fp)
        return params
