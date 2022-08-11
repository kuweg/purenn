import numpy as np


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray):
    return -np.sum(y_true * np.log(y_pred + 10**-100))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

def BinaryCrossEntropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


class MeanSquaredError:
    
    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
        return 2*(y_pred-y_true)/y_true.size
    
    __call__ = calc
    

