from abc import abstractmethod
from sklearn.metrics import roc_auc_score

class Model:
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y, sample_weight):
        predictions = self.predict(X)
        return roc_auc_score(y, predictions, sample_weight=sample_weight)

