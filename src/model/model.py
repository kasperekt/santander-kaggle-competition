from abc import abstractmethod


class Model:
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
