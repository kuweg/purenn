from __future__ import annotations

from itertools import pairwise
from typing import Any, Callable, Iterable, List
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


def flatten_list(lst: List[List]) -> List[Any]:
    return [item for sublist in lst for item in sublist]

 
def dummy_callable(*args: Any) -> Any:
    return args


vertical_char = '|'
bottom_char = '-'
corner_char = '+'
max_line_len = 12

def add_vertical_row(table: 'PrettyTable', row_name: str, row_value: str):
    n_cells = len(table.field_names)
    total_line_length = sum(
        list(
            map(
                lambda field: len(field)+2, table.field_names
                )
            )
        ) + n_cells + 1
    if len(row_name) + 2 < max_line_len:
        row_name = row_name.center(max_line_len - 1, ' ')
    opt_line = vertical_char + ' ' + row_name + vertical_char +' ' + row_value
    e_space = total_line_length - len(opt_line) - 1
    opt_line += ' ' * e_space
    opt_line += vertical_char
    empty_bottom_space = total_line_length - 2
    bottom_line = corner_char + '-'*empty_bottom_space + corner_char
    return opt_line, bottom_line