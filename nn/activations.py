from typing import Callable
import numpy as np

from .utils import set_repr


def apply_function_to_nparray(array: np.ndarray, fn: Callable) -> np.ndarray:
    initial_shape = array.shape
    mapped_array = np.array(
        list(map(fn, array.flat))
    )
    return mapped_array.reshape(initial_shape)


@set_repr('relu')
def relu(array: np.ndarray, derivative: bool=False) -> float:
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if derivative:
        
        return apply_function_to_nparray(
            array, lambda value: 1. if value >=0 else 0.
        )
    return np.maximum(array, 0.)

    
@set_repr('tanh')    
def tanh(array: np.ndarray, derivative: bool=False) -> np.ndarray:
        
    if derivative:
        return 1 - np.tanh(array)**2

    return np.tanh(array)


@set_repr('sigmoid')
def sigmoid(array: np.ndarray, derivative: bool=False) -> np.ndarray:
    if derivative:
        return sigmoid_derivative(array)
    return 1 / (1 + np.exp(-array))


def sigmoid_derivative(array: np.ndarray) -> np.ndarray:
    s = sigmoid(array)
    return s * (1 - s)