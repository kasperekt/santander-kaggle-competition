import numpy as np

from config import use_gpu
from catboost import CatBoostClassifier
from model.explorer import Explorer


class CatboostExplorer(Explorer):
    def __init__(self, **kwargs):
        super(CatboostExplorer, self).__init__('catboost', **kwargs)

    def get_predictions(self, X_train, y_train, eval_set, **params):
        model = get_model(**params)
        model.fit(X_train, y_train, eval_set=eval_set)

        X_val = eval_set[0]
        predictions = np.array(model.predict_proba(X_val))
        predictions = predictions[:, 1]

        return predictions


def get_model(gpu_force=False, **params):
    if 'learning_rate' not in params:
        params['learning_rate'] = 0.03

    if 'loss_function' not in params:
        params['loss_function'] = 'Logloss'

    if 'depth' not in params:
        params['depth'] = 3

    return CatBoostClassifier(**params,
                              use_best_model=True,
                              eval_metric='AUC',
                              task_type='GPU' if (gpu_force or use_gpu) else 'CPU',
                              logging_level='Verbose')

