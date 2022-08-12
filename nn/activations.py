from typing import Callable
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
