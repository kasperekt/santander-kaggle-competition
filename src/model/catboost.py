import numpy as np

from config import use_gpu
from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid
from model.explorer import Explorer
from model.model import Model


class CatboostExplorer(Explorer):
    def __init__(self, **kwargs):
        super(CatboostExplorer, self).__init__('catboost', **kwargs)
        self.param_grid = ParameterGrid({'learning_rate': [0.03],
                                         'iterations': [5000],
                                         'l2_leaf_reg': np.logspace(-19, -15, 10),
                                         'border_count': [254],
                                         'depth': [2, 3, 4, 5, 6]})

    def get_predictions(self, dataset, **params):
        model = CatboostModel(**params)
        model.fit(dataset)

        predictions = model.predict(dataset.X_val)

        return predictions


class CatboostModel(Model):
    def __init__(self, **params):
        if 'learning_rate' not in params:
            params['learning_rate'] = 0.03

        if 'loss_function' not in params:
            params['loss_function'] = 'Logloss'

        if 'depth' not in params:
            params['depth'] = 3

        super(CatboostModel, self).__init__(**params)

        task_type = 'GPU' if use_gpu else 'CPU'
        self.model = CatBoostClassifier(**params,
                                        use_best_model=True,
                                        eval_metric='AUC',
                                        task_type=task_type,
                                        logging_level='Verbose')

    def fit(self, dataset):
        eval_set = (dataset.X_val, dataset.y_val)
        self.model.fit(dataset.X_train, dataset.y_train, eval_set=eval_set)

    def predict(self, X):
        predictions = np.array(self.model.predict_proba(X), dtype=np.float)
        predictions = predictions[:, 1]
        return predictions

