import numpy as np
from typing import Any, Callable
from numbers import Number


def _shape_checking(*args) -> Number | tuple:
    if len(args) == 1:
        return int(args[0])
    return tuple(map(int, args))


def init_bias(*args) -> np.ndarray:
    bias_shape = args[0]
    if len(args) == 2:
        bias_shape = args[1]
    return np.random.rand(1, bias_shape)


class Weights:
    
    def __init__(self, weights_strategy: str=None) -> None:
        self.weights_strategy = weights_strategy
        
    # TODO: xavier weigths initialization

    @staticmethod
    def zeros(shape: int | tuple) -> np.ndarray:
        return np.zeros(shape)
    
    @staticmethod
    def ones(shape: int | tuple) -> np.ndarray:
        return np.ones(shape)

    @staticmethod
    def rand(shape: int | tuple, scaling: float=.01) -> np.ndarray:

        if isinstance(shape, tuple):
            return np.random.rand(*shape) * scaling
        return np.random.rand(shape) * scaling
    
    def get_weights_initializator(self) -> Callable:
        
        if hasattr(self, self.weights_strategy):
            return object.__getattribute__(self, self.weights_strategy)
        raise AttributeError('Specified weights strategy is not exists.')
    
    def __call__(self, *args: Any) -> Any:
        shape = _shape_checking(*args)
        return self.get_weights_initializator()(shape)