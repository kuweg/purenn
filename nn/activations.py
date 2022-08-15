from typing import Callable, Union
import numpy as np

from .utils import set_repr


@set_repr('relu')
def relu(array: np.ndarray, derivative: bool=False) -> float:
    
    if derivative:
        return (array > 0) * 1.
    return np.maximum(array, 0.)

    
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



class ActivationExistsError(Exception):
    pass


ACTIVATION_MAP ={
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid
}

def get_mapped_function(function_name: str) -> Union[Callable, None]:
    
    if function_name in ACTIVATION_MAP.keys():
        return ACTIVATION_MAP[function_name]
    raise ActivationExistsError(
        'Activation function {} is not exists. Please, use one of the {}'.format(
            function_name, list(ACTIVATION_MAP.values())
        )
    )


def set_activation(activation: Union[Callable, str]) -> Callable:
    return (activation if isinstance(activation, Callable)
            else get_mapped_function(activation))
