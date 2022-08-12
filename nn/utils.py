from itertools import pairwise
from typing import Any, Callable, Iterable
import numpy as np


class rewrapper:
    
    def __init__(self, repr: str, func: Callable) -> None:
        self._repr = repr
        self._func = func
        
    def __call__(self, *args: Any, **kwds: Any) -> Callable:
        return self._func(*args, **kwds)
    
    def __repr__(self) -> str:
        return self._repr
    

def set_repr(repr_name: str) -> Callable:
    def _wrap(func: Callable) -> rewrapper:
        return rewrapper(repr_name, func)
    return _wrap


def reversed_pairwise(sequence: Iterable) -> list[list]:
    pairwise_sequence = list(pairwise(sequence))
    reversed_pairwise_sequence = list(
        map(
            lambda pair: list(reversed(pair)),
            pairwise_sequence
            )
        )
    return reversed_pairwise_sequence


def apply_function_to_nparray(array: np.ndarray, fn: Callable) -> np.ndarray:
    initial_shape = array.shape
    mapped_array = np.array(
        list(map(fn, array.flat))
    )
    return mapped_array.reshape(initial_shape)

