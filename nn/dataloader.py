from __future__ import annotations

import numpy as np
from typing import Callable, Generator, Iterable, Tuple, List, Union

from nn.utils import dummy_callable


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
        [
            np.array([x_i, y_i], dtype=object)
            for x_i, y_i in zip(x,y)],
        dtype=object
    )


def _random_from_array(array: np.ndarray) -> Union[int, int]:
    random_pick_idx = np.random.choice(range(array.shape[0]))
    
    return random_pick_idx

def make_batches(x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int) -> Tuple[np.ndarray, np.ndarray]:

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    x0, x1 = x.shape
    new_x0, new_x1 = int(x0 / batch_size), x1
    x_batch = x.reshape(new_x0, batch_size, new_x1)
    
    y0, y1 = y.shape[0], y.shape[1]
    new_y0 = int(y0 / batch_size)
    y_batch = y.reshape(new_y0, batch_size, y1)

    return [x_batch, y_batch]


def _input_fit_handler(X_train: Union[np.ndarray, DataLoader],
                       y_train: np.ndarray=None) -> bool:
    if isinstance(X_train, DataLoader) and y_train is None:
        print('{} type as input data'.format(type(X_train)))
        return X_train.data
    elif isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
        print('{} and {} types as input data'.format(type(X_train), type(y_train)))
        return [(x, y) for x, y in zip(X_train, y_train)]
    else:
        raise AttributeError(
            'When passing a DataLoader object instead of separate x,y arguments\n' +
            ' make sure that y is None\n'
        )


class DataLoader:
    
    def __init__(self,
                 X_input: np.ndarray,
                 y_input: np.ndarray) -> None:

        if _check_shape_equality(X_input, y_input):
            self.x = np.array(X_input)
            self.y = np.array(y_input)
            self._data = [self.x, self.y]
            self.bs = None
        
    def apply(self,
              to_x: bool=None,
              to_y: bool=None,
              both: bool=None,
              fn: Union[Callable, List[Callable]]=dummy_callable) -> None:

        if both: 
            self._data = np.array(
                [fn(*xy_i) for xy_i in zip(self._data[0], self._data[1])]
                )

        if to_x:
            self.x = fn(self.x)

        if to_y:
            self.y = fn(self.y)
            
        self._data = [self.x, self.y]
        
        return self
        
    def batch(self, batch_size: int,
              upsample: bool=False,
              loop: bool=False) -> DataLoader:
        self.bs = batch_size
        n_samples = self.x.shape[0]
        n_batches = int(n_samples // batch_size)
        x = self.x.copy()
        y = self.y.copy()
        if len(y.shape) in [0, 1]:
            y = y.reshape(-1, 1)
        
        print('Recieved shapes:')
        print(x.shape, y.shape)
        if (n_batches * batch_size < n_samples) and upsample:
            
            n_batches +=1
            n_missing = (n_batches * batch_size) - n_samples
            print('Need to upsample:', n_missing)
            upsample_values_x = np.empty_like(self.x[0])
            upsample_values_y = np.empty_like(self.y[0])
            
            for _ in range(n_missing):
                sample_index = _random_from_array(self.x)
                upsample_values_x = np.vstack(
                    (
                        upsample_values_x,
                        x[sample_index]
                    )
                )
                upsample_values_y = np.vstack(
                    (
                        upsample_values_y,
                        y[sample_index]
                    )
                )
            upsample_values_x = upsample_values_x[1:]
            upsample_values_y = upsample_values_y[1:]
            
            y = np.vstack((upsample_values_y, y))
            x = np.vstack((x, upsample_values_x))

        elif not upsample:
            n_extra_values = n_samples - (n_batches * batch_size)
            print('Will be removed:', n_extra_values)
            x_extra_ids = []
            y_extra_ids = []
            for _ in range(n_extra_values):
                sample_index = _random_from_array(x)
                x_extra_ids.append(sample_index)
                y_extra_ids.append(sample_index)
            
            x = np.delete(x, x_extra_ids, axis=0)
            y = np.delete(y, y_extra_ids, axis=0)
                
        

        self._data = make_batches(x, y, batch_size)

        return self
    
    @property
    def data(self) -> Iterable[np.ndarray, np.ndarray]:
        return  np.array(
                    [
                        (np.transpose(x), np.transpose(y)) for
                        x, y in zip(self._data[0], self._data[1])
                        ],
                    dtype=object 
                )
        
        
        
    