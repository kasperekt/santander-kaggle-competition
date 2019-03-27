import lightgbm as lgb
import numpy as np

from model.explorer import Explorer
from model.model import Model
from sklearn.model_selection import ParameterGrid
from config import use_gpu


class LightGBMExplorer(Explorer):
    def __init__(self, num_round=10, **kwargs):
        super(LightGBMExplorer, self).__init__("lightgbm", **kwargs)
        self.num_round = num_round
        self.param_grid = ParameterGrid({'num_leaves': np.linspace(30, 255, 3, dtype=np.int),
                                         'max_depth': [10, 15, 20],
                                         'min_data_in_leaf': np.linspace(10, 100, 3, dtype=np.int),
                                         'max_bin': np.linspace(100, 512, 3, dtype=np.int),
                                         'feature_fraction': np.linspace(0.5, 1, 3),
                                         'num_iterations': [2500],
                                         'learning_rate': [0.03],
                                         'objective': ['binary'],
                                         'metric': ['auc']})

    def get_predictions(self, dataset, **params):
        if use_gpu:
            params['device'] = 'gpu'

        model = lgb.train(params,
                          dataset.train_data,
                          self.num_round,
                          valid_sets=[dataset.val_data],
                          verbose_eval=500,
                          early_stopping_rounds=100)

        train_predictions = model.predict(dataset.X_train, num_iteration=model.best_iteration)
        val_predictions = model.predict(dataset.X_val, num_iteration=model.best_iteration)

        return train_predictions, val_predictions


class LightGBMModel(Model):
    def __init__(self, num_round=10, **params):
        super(LightGBMModel, self).__init__(**params)
        self.num_round = num_round
        self.model = None

    def fit(self, dataset):
        self.model = lgb.train(self.params, dataset.train_data, self.num_round, valid_sets=[dataset.val_data])

    def predict(self, X):
        self.model.predict(X, num_iteration=self.model.best_iteration)

