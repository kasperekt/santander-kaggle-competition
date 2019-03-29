import pickle

from model.catboost import CatboostExplorer
from model.lightgbm import LightGBMExplorer
from utils import get_params_filepath


def pick_explorer(explorer_type):
    if explorer_type == 'catboost':
        return CatboostExplorer()
    elif explorer_type == 'lightgbm':
        return LightGBMExplorer()
    else:
        raise ValueError('You\'ve passed wrong explorer type ({})'.format(explorer_type))


def find_params(dataset, explorer_type):
    explorer = pick_explorer(explorer_type)
    params = explorer.fit(dataset)
    return params


def load_params(explorer_type):
    params_file = get_params_filepath(explorer_type)

    with open(params_file, 'rb') as params_fp:
        params = pickle.load(params_fp)
        return params
