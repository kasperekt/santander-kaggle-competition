import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import current_date_str
from config import OUT_DIR
from abc import abstractmethod


class Explorer:
    def __init__(self, name, param_grid, test_size=0.1):
        self.name = name
        self.param_grid = param_grid
        self.test_size = test_size

    def fit(self, X, y, eval_set=None):
        params_file = os.path.join(OUT_DIR, 'params.{}.pickle'.format(self.name))

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
            predictions = self.get_predictions(X_train, y_train, eval_set=(X_val, y_val), **params)
            score = roc_auc_score(y_val, predictions)
            print('Score: {}'.format(score))

            result = {**params, 'score': score}
            results.append(result)

            if score > best_score:
                best_params = params
                best_score = score
                with open(params_file, 'wb') as params_fp:
                    pickle.dump(best_params, params_fp)

        results_filename = os.path.join(OUT_DIR, 'results.{}.csv'.format(current_date_str()))
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_filename, sep=",")

        return best_params

    @abstractmethod
    def get_predictions(self, X_train, y_train, eval_set, **params):
        raise NotImplementedError('You need to implement this method.')
