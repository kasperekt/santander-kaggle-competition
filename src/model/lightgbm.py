import lightgbm as lgb

from model.explorer import Explorer
from model.model import Model
from sklearn.model_selection import ParameterGrid


class LightGBMExplorer(Explorer):
    def __init__(self, num_round=10, **kwargs):
        super(LightGBMExplorer, self).__init__("lightgbm", **kwargs)
        self.num_round = num_round
        self.param_grid = ParameterGrid({'num_leaves': [12],
                                         'num_trees': [10],
                                         'objective': ['binary'],
                                         'metric': ['auc']})

    def get_predictions(self, dataset, **params):
        model = lgb.train(params, dataset.train_data, self.num_round, valid_sets=[dataset.val_data])
        predictions = model.predict(dataset.X_val, num_iteration=model.best_iteration)
        return predictions


class LightGBMModel(Model):
    def __init__(self, num_round=10, **params):
        super(LightGBMModel, self).__init__(**params)
        self.num_round = num_round
        self.model = None

    def fit(self, dataset):
        self.model = lgb.train(self.params, dataset.train_data, self.num_round, valid_sets=[dataset.val_data])

    def predict(self, X):
        self.model.predict(X, num_iteration=self.model.best_iteration)

