from abc import ABC, abstractmethod
from typing import Callable, Union
import numpy as np

from .exceptions import UnknownActivationError
from .utils import set_repr


class Activation(ABC):

    @abstractmethod
    def activation(array: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def df(array: np.ndarray) -> np.ndarray:
        pass
    
    def __call__(self, array: np.ndarray) -> Callable:
        return self.activation(array)
    
    def __repr__(self) -> str:
        return self.__class__.__name__.lower()


class Relu(Activation):
    
    def activation(self, array: np.ndarray) -> np.ndarray:
        return np.maximum(array, 0.)
    
    def df(self, array: np.ndarray) -> np.ndarray:
        return (array > 0) * 1.


class LeakyRelu(Activation):
    
    def activation(self, array: np.ndarray, alpha: float=0.01) -> np.ndarray:
        return np.maximum(array, alpha)
    
    def df(self, array: np.ndarray, alpha: float=0.01) -> np.ndarray:
        return np.where(array<=0, alpha, 1.) 
    
    def __repr__(self) -> str:
        return 'leaky_relu'


class Tanh(Activation):
    
    def activation(self, array: np.ndarray) -> np.ndarray:
        return np.tanh(array)
    
    def df(self, array: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(array)**2


class Sigmoid(Activation):
    
    def activation(self, array: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-array)) 
    
    def df(self, array: np.ndarray) -> np.ndarray:
        return self.activation(array) * (1 - self.activation(array))


class Softmax(Activation):
    
    def activation(self, array: np.ndarray, stable: bool=False) -> np.ndarray:
        if stable:
            array = array - np.max(array)
        return np.exp(array) / np.sum(np.exp(array), axis=0)
    
    def df(self, array: np.ndarray) -> np.ndarray:
        I = np.eye(array.shape[0])
        return self.activation(array) * (I - self.activation(array).T) 

# @set_repr('softmax')
# def softmax(array: np.ndarray) -> np.ndarray:
#     shifted_array = array - np.max(array)
#     numerator = np.exp(shifted_array)
#     denominator = np.sum(numerator, axis=0)
#     return numerator/denominator


relu = Relu()
tanh = Tanh()
sigmoid = Sigmoid()
leaky_relu = LeakyRelu()
softmax = Softmax()

ACTIVATION_MAP ={
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'softmax': softmax,
    'leaky_relu': leaky_relu
}


def get_mapped_function(function_name: str) -> Union[Callable, None]:
    
    if function_name in ACTIVATION_MAP.keys():
        return ACTIVATION_MAP[function_name]
    raise UnknownActivationError(
        'Activation function {} is not exists. Please, use one of these {}'.format(
            function_name, list(ACTIVATION_MAP.values())
        )
    )


def set_activation(activation: Union[Callable, str]) -> Callable:
    return (activation if isinstance(activation, Callable)
            else get_mapped_function(activation))
    

def is_softmax(activation_fn: Callable) -> bool:
    return activation_fn is softmax
