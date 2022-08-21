from abc import ABC, abstractstaticmethod
import numpy as np
import re


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray):
    return -np.sum(y_true * np.log(y_pred + 10**-100))


def BinaryCrossEntropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


class Loss(ABC):
    
    @abstractstaticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass
    
    @abstractstaticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass
    
    def __repr__(self) -> str:
        splitted_name = '_'.join(
            re.findall('.[^A-Z]*', self.__class__.__name__)
            )
        return splitted_name.lower()

class MeanSquaredError(Loss):
    
    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
        return 2*(y_pred-y_true)/y_true.size
    
    __call__ = calc
