import lightgbm as lgb
import numpy as np

from model.explorer import Explorer
from model.model import Model
from sklearn.model_selection import ParameterGrid
from config import use_gpu


class LightGBMExplorer(Explorer):
    def __init__(self, **kwargs):
        super(LightGBMExplorer, self).__init__("lightgbm", **kwargs)
        self.__estimator = None
        self.param_grid = ParameterGrid({'num_leaves': [24, 32],
                                         'min_data_in_leaf': [500],
                                         'max_bin': [64, 128],
                                         'feature_fraction': [0.2],
                                         'max_depth': [20],
                                         'reg_alpha': [1.5, 2.5],
                                         'reg_lambda': [1.5, 2.5]})

    def get_predictions(self, dataset, **params):
        if use_gpu:
            params['device'] = 'gpu'

        model = LightGBMModel(**params)
        model.fit(dataset)

        train_predictions = model.predict(dataset.X_train)
        val_predictions = model.predict(dataset.X_val)

        return train_predictions, val_predictions


class LightGBMModel(Model):
    def __init__(self, **params):
        if 'num_iterations' not in params:
            params['num_iterations'] = 5000

        if 'learning_rate' not in params:
            params['learning_rate'] = 0.01

        super(LightGBMModel, self).__init__(**params)

        self.model = lgb.LGBMClassifier(n_jobs=4,
                                        metric='auc',
                                        objective='binary',
                                        **params)

    def fit(self, dataset):
        eval_set = (dataset.X_val, dataset.y_val)
        self.model.fit(dataset.X_train, dataset.y_train, eval_set=eval_set, early_stopping_rounds=500, verbose=100)

    def predict(self, X):
        predictions = self.model.predict_proba(X, num_iteration=self.model.best_iteration_)
        predictions = np.array(predictions, dtype=np.float)[:, 1]
        return predictions

