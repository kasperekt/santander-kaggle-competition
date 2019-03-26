import os
import pickle
import pandas as pd

from utils import current_date_str
from config import use_gpu, OUT_DIR
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


class CatboostExplorer:
    name = 'catboost'

    def __init__(self, param_grid, test_size=0.1):
        self.param_grid = param_grid
        self.test_size = test_size

    def fit(self, X, y, eval_set=None, params_file='../out/params.catboost.pickle'):
        if eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        else:
            X_train = X
            y_train = y
            X_val, y_val = eval_set

        best_params = None
        best_score = 0

        results = []

        for params in self.param_grid:
            print('Testing params: ', params)

            model = get_model(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            score = model.score(X_val, y_val)

            print('Score: {}'.format(score))

            result = {**params, 'score': score}
            results.append(result)

            if score > best_score:
                best_params = params
                with open(params_file, 'wb') as params_fp:
                    pickle.dump(best_params, params_fp)

        results_filename = os.path.join(OUT_DIR, 'results.{}.csv'.format(current_date_str()))
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_filename, sep=",")

        return best_params


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

