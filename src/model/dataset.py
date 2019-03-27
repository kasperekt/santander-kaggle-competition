import lightgbm as lgb

from data import get_data

class Dataset:
    def __init__(self, model='catboost', force_reload=False, strategy='oversampling'):
        X_train, X_val, y_train, y_val = get_data(force_reload=force_reload, strategy=strategy)

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        if model == 'lightgbm':
            self.train_data = lgb.Dataset(X_train, label=y_train)
            self.val_data = lgb.Dataset(X_val, label=y_val)

    def get_X_train(self):
        return self.X_train

    def get_X_val(self):
        return self.X_val

    def get_y_train(self):
        return self.y_train

    def get_y_val(self):
        return self.y_val


