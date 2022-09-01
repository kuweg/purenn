from typing import Callable, Union
import numpy as np

from .exceptions import UnknownActivationError
from .utils import set_repr


@set_repr('relu')
def relu(array: np.ndarray, derivative: bool=False) -> float:
    
    if derivative:
        return (array > 0) * 1.
    return np.maximum(array, 0.)


@set_repr('leaky_relu')
def leaky_relu(array: np.ndarray,
               derivative: bool=False,
               alpha: float=0.0001):
    
    if derivative:
        return np.where(array<=0, alpha, 1) 
    return np.maximum(array, alpha) 


@set_repr('tanh')    
def tanh(array: np.ndarray, derivative: bool=False) -> np.ndarray:
        
    if derivative:
        return 1 - np.tanh(array)**2

    return np.tanh(array)


@set_repr('sigmoid')
def sigmoid(array: np.ndarray, derivative: bool=False) -> np.ndarray:
    
    g = 1 / (1 + np.exp(-array)) 
    
    if derivative:
        g = g * (1 - g) 
    return g


@set_repr('softmax')
def softmax(array: np.ndarray) -> np.ndarray:
    shifted_array = array - np.max(array)
    numerator = np.exp(shifted_array)
    denominator = np.sum(numerator)
    return numerator/denominator


@set_repr('softmax2')
def softmax2(array: np.ndarray, derivative: bool=False) -> np.ndarray:
    
    shifted_array = array - np.max(array)
    numerator = np.exp(shifted_array)
    denominator = np.sum(numerator)
    softmax_ = numerator/denominator
    if derivative:
        softmax(1 - softmax_)
        
    return softmax_



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
        'Activation function {} is not exists. Please, use one of the {}'.format(
            function_name, list(ACTIVATION_MAP.values())
        )
    )


def set_activation(activation: Union[Callable, str]) -> Callable:
    return (activation if isinstance(activation, Callable)
            else get_mapped_function(activation))
    
def is_softmax(activation_fn: Callable) -> bool:
    return activation_fn is softmax
