import numpy as np
from typing import Callable, Tuple, Union
from numbers import Number


class InputShapeError(Exception):
    pass


def _check_shape_equality(x: Union[np.ndarray, list],
                          y: Union[np.ndarray, list]
                          ) -> Union[bool, None]:
    if x.shape[0] == y.shape[0]:
        return True
    else:
        raise InputShapeError(
            "Amount of X samples shoulbe same as y (x = y)."
        )
        
def _form_pairs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array(
        [(x_i, y_i) for x_i, y_i in zip(x,y)]
    )


def _random_from_array(array: np.ndarray) -> Union[np.ndarray, Number]:
    donor_index = np.random.choice(range(array.shape[0]))
    
    return array[donor_index]


class DataLoader:
    
    def __init__(self,
                 X_input: np.ndarray,
                 y_input: np.ndarray) -> None:

        if _check_shape_equality(X_input, y_input):
            self.x = np.array(X_input)
            self.y = np.array(y_input)
            self.data = _form_pairs(self.x, self.y)
            self.n = self.data.shape[0]
        
    def apply(self, to_x: bool, to_y: bool, fn: Callable) -> None:

        if to_x and to_y:
            self.data = np.array(
                [fn(*xy_i) for xy_i in self.data]
            )

        if to_x:
            self.data = np.array(
                [fn(x_i) for x_i, _ in self.data]
            )

        elif to_y:
            self.data = np.array(
                [fn(y_i) for _, y_i in self.data]
            )
        
        
    def batch(self, batch_size: int, loop: bool=False):
        init_shape = self.data.shape
        print(init_shape)
        n_samples = init_shape[0]
        sample_volume = init_shape[1]
        n_batches = int(n_samples // batch_size)
        
        if n_batches * batch_size < n_samples:
            n_batches +=1
            n_missing = (n_batches * batch_size) - n_samples
            print('Need to upsample:', n_missing)
            upsample_values = np.empty_like(self.data[0])
            for _ in range(n_missing):
                upsample_values = np.vstack(
                    (
                        upsample_values,
                        _random_from_array(self.data)
                    )
                )
            upsample_values = upsample_values[1:]
            self.data = np.vstack((self.data, upsample_values))
            
        self.data = self.data.reshape(n_batches, batch_size, sample_volume)
        
    