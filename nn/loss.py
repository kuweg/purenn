from abc import ABC, abstractstaticmethod
from typing import Callable
import numpy as np

class Loss(ABC):
    
    @abstractstaticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass
    
    @abstractstaticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Callable:
        return self.calc(y_true, y_pred)
    
    def __repr__(self) -> str:
        return self.__class__.__name__

class MeanSquaredError(Loss):
    
    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
        return 2*(y_pred-y_true)/y_true.size
  

class MeanAbsoluteError(Loss):
    
    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return np.mean(abs(y_true - y_pred))
    
    @staticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        n = y_true.shape[0]
        return -((y_true - y_pred) / (abs(y_true - y_pred) + 10**-100))/n


class CategoricalCrossEntropy(Loss):
    """
    This function calculates and return the categorical-crossentropy-loss.
    "+1e-15" is just for adding a very small number to avoid np.log(0).
    
    :param y_true: the current predicted output of the model
    :type y_true: np.ndarray
    :param y_pred: the expected output
    :type y_rped: np.ndarray
    :return: the categorical-crossentropy-loss
    :rtype: np.float64
    """
    
    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        loss = -np.sum(y_true * (np.log(y_pred+1e-10)))
        return loss / float(y_pred.shape[0])
    
    @staticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return y_pred - y_true


class CrossEntropyLoss(Loss):

    @staticmethod
    def calc(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        loss = -np.sum(y_true * np.log(y_pred + 10**-100))
        return loss
    
    @staticmethod
    def df(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return -y_true/(y_pred + 10**-100)
