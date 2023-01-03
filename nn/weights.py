import numpy as np
from typing import Any, Callable, Union
from numbers import Number


def _shape_checking(*args) -> Union[Number, tuple]:
    if len(args) == 1:
        return int(args[0])
    return tuple(map(int, args))

class Weights:
    
    def __init__(self, weights_strategy: str=None) -> None:
        self.weights_strategy = weights_strategy
        
    @staticmethod
    def xavier(shape: tuple) -> np.ndarray:
        return np.random.randn(*shape) / np.sqrt(shape[0])

    @staticmethod
    def zeros(shape: Union[int, tuple]) -> np.ndarray:
        return np.zeros(shape)
    
    @staticmethod
    def ones(shape: Union[int, tuple]) -> np.ndarray:
        return np.ones(shape)

    @staticmethod
    def rand(shape: Union[int, tuple]) -> np.ndarray:

        if isinstance(shape, tuple):
            return np.random.randn(*shape)
        return np.random.randn(shape)

    @property
    def _strategies(self) -> list[str]:
        return [strat for strat in list(self.__class__.__dict__)
                if not strat.startswith('_')]
    
    def _get_weights_initializator(self) -> Callable:
        
        if hasattr(self, self.weights_strategy):
            return object.__getattribute__(self, self.weights_strategy)
        raise AttributeError('Specified weights strategy is not exists.'+
                             'Please, use one of these {} strategies'
                             .format(self._strategies))
    
    def __call__(self, *args: Any) -> Any:
        shape = _shape_checking(*args)
        return self._get_weights_initializator()(shape)