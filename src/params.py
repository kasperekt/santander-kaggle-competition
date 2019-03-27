import pickle
import numpy as np

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from model.catboost import get_model, CatboostExplorer
from utils import get_params_filepath


def pick_explorer(explorer_type, param_grid):
    if explorer_type == 'catboost':
        return CatboostExplorer(param_grid=param_grid)
    else:
        raise ValueError('You\'ve passed wrong explorer type ({})'.format(explorer_type))


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


def find_params(X, y, explorer_type, eval_set=None):
    param_grid = ParameterGrid({'learning_rate': [0.03],
                                'iterations': [5000],
                                'l2_leaf_reg': np.logspace(-19, -15, 10),
                                'border_count': [254],
                                'depth': [2, 3, 4, 5, 6]})

    explorer = pick_explorer(explorer_type, param_grid)
    params = explorer.fit(X, y, eval_set=eval_set)
    return params


def load_params(explorer_type):
    params_file = get_params_filepath(explorer_type)

    with open(params_file, 'rb') as params_fp:
        params = pickle.load(params_fp)
        return params
