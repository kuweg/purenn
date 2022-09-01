from typing import Union
import numpy as np


def categorical_encoding(array: Union[np.ndarray, list]) -> np.ndarray:
    print(array.shape)
    len_ = np.max(array) + 1
    initial_shape = array.flatten().shape
    new_shape = (initial_shape[0], len_)
    
    encoded_array = np.empty(new_shape)
    
    for value, row in zip(array, range(initial_shape[0])):
        sample = np.zeros(len_)
        sample[value] = 1.
        
        encoded_array[row] = np.expand_dims(sample, axis=0)
        
    return encoded_array


def transform_input_data(array: np.ndarray) -> np.ndarray:
    return np.array(
        list(
            map(
                lambda sample: sample.reshape(-1, 1), array
                )
            )
        )