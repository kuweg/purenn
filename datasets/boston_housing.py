from typing import Tuple
import numpy as np
import pandas as pd


FILE_URL = 'http://lib.stat.cmu.edu/datasets/boston'
HEADERS = ['CRIM', 'ZN', 'IDIUS', 'CHAS', 'NOX', 'RM',
           'AGE', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B',
           'LSTAT', 'MEDV']


def download_boston_housing() -> Tuple[np.ndarray, np.ndarray]:
    raw_df = pd.read_csv(FILE_URL, sep='\s+', skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target


class BostonHousing:
    """
    Boston housing dataset loader.
    Dataset stored in a few different variations:
    `data` - raw numpy values, which can be splitted using
             test_size variable.
    `df` - a pandas dataframe for exploration and analysis.
    :param test_size: size of data splittng.
                       Default values is None ->
                       returns tuple of (x_data, y_target) 
    :type test_size: float
    """
    
    def __init__(self, test_size: float=0.) -> None:
        assert 0.<= test_size <= 1.
        self.split_size = test_size
        self._data = download_boston_housing()
        
    @property
    def data(self):
        if self.split_size > 0.:
            data_len = self._data[0].shape[0]
            test_split = int(data_len * self.split_size)
            train_split = int(data_len - test_split)
            x_train, y_train = (self._data[0][:train_split],
                                self._data[1][:train_split])
            x_test, y_test = (self._data[0][train_split:],
                              self._data[1][train_split:])
            return (x_train, y_train), (x_test, y_test)
        return self._data
    
    @property
    def df(self):
        full_data = np.hstack((self._data[0], self._data[1].reshape(-1, 1)))
        df = pd.DataFrame(full_data, columns=HEADERS)
        return df
        