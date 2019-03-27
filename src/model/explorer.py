import os
import pandas as pd
import pickle
import sys

from sklearn.metrics import roc_auc_score
from utils import current_date_str
from config import OUT_DIR
from abc import abstractmethod


class Explorer:
    def __init__(self, name, test_size=0.1):
        self.name = name
        self.test_size = test_size

    def fit(self, dataset):
        params_file = os.path.join(OUT_DIR, 'params.{}.pickle'.format(self.name))
        best_params = None
        best_score = 0

        results = []

        for params in self.param_grid:
            print('Testing params: ', params)
            train_predictions, val_predictions = self.get_predictions(dataset, **params)
            train_score = roc_auc_score(dataset.y_train, train_predictions)
            val_score = roc_auc_score(dataset.y_val, val_predictions)
            print('Score: Train = {}; Val={}'.format(train_score, val_score))

            result = {**params, 'train_score': train_score, 'score': val_score}
            results.append(result)

            if val_score > best_score:
                best_params = params
                best_score = val_score
                with open(params_file, 'wb') as params_fp:
                    pickle.dump(best_params, params_fp)

        sys.stdout.flush()

        results_filename = os.path.join(OUT_DIR, 'results.{}.{}.csv'.format(self.name, current_date_str()))
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_filename, sep=",")

        return best_params

    @abstractmethod
    def get_predictions(self, dataset, **params):
        raise NotImplementedError('You need to implement this method.')
