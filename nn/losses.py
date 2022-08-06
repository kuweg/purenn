import numpy as np


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray):
    return -np.sum(y_true * np.log(y_pred + 10**-100))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    error_sum = 0
    for ac, ex in zip(y_pred, y_true):
        error_sum += (ac - ex)**2
    return error_sum / len(y_pred)

def BinaryCrossEntropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()