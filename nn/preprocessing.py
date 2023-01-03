from typing import Union
import numpy as np


def categorical_encoding(array: Union[np.ndarray, list]) -> np.ndarray:
    len_ = np.max(array) + 1
    initial_shape = array.flatten().shape
    new_shape = (initial_shape[0], len_)
    
    encoded_array = np.empty(new_shape)
    
    for value, row in zip(array, range(initial_shape[0])):
        sample = np.ones(len_) * 0.01
        sample[value] = 0.99
        
        encoded_array[row] = np.expand_dims(sample, axis=0)
        
    return encoded_array


def transform_input_data(array: np.ndarray) -> np.ndarray:
    """
    Transforming data from `(N, )` to `(N, 1)`.
    """
    return np.array(
        list(
            map(
                lambda sample: sample.reshape(-1, 1), array
                )
            )
        )
    

def normalize(data: np.ndarray) -> np.ndarray:
    """
    Scale input data in [0:1]
    
    :param data: np.ndarray with numeric data
    :type data: np.ndarray
    :return: scaled arrray with values [0:1]
    :rtype: np.ndarray
    """
    data = data
    scaled_data = (data / np.max(data) * 0.99) + 0.01
    return scaled_data


def preprocessing_pipeline(data: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline.
    `Normalization -> shape transformation`
    """
    
    norm_data = normalize(data)
    transformed_data = transform_input_data(norm_data)
    return transformed_data