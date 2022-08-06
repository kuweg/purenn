import functools
from typing import Any, Callable

class rewrapper:
    
    def __init__(self, repr: str, func: Callable) -> None:
        self._repr = repr
        self._func = func
        
    def __call__(self, *args: Any, **kwds: Any) -> Callable:
        return self._func(*args, **kwds)
    
    def __repr__(self) -> str:
        return self._repr
    

def set_repr(reprfunc: Callable) -> Callable:
    def _wrap(func: Callable) -> rewrapper:
        return rewrapper(reprfunc, func)
    return _wrap