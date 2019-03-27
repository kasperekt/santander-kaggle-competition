import pickle
import numpy as np

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from model.catboost import CatboostExplorer
from model.lightgbm import LightGBMExplorer
from utils import get_params_filepath


def pick_explorer(explorer_type, param_grid):
    if explorer_type == 'catboost':
        return CatboostExplorer()
    elif explorer_type == 'lightgbm':
        return LightGBMExplorer(num_round=1)
    else:
        raise ValueError('You\'ve passed wrong explorer type ({})'.format(explorer_type))


def find_params(dataset, explorer_type):
    param_grid = ParameterGrid({'learning_rate': [0.03],
                                'iterations': [5000],
                                'l2_leaf_reg': np.logspace(-19, -15, 10),
                                'border_count': [254],
                                'depth': [2, 3, 4, 5, 6]})

    explorer = pick_explorer(explorer_type, param_grid)
    params = explorer.fit(dataset)
    return params


def load_params(explorer_type):
    params_file = get_params_filepath(explorer_type)

    with open(params_file, 'rb') as params_fp:
        params = pickle.load(params_fp)
        return params
