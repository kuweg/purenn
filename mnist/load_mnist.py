from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

def check_and_load(filepath: str) -> pd.DataFrame:
    if os.path.isfile(filepath):
        return pd.read_csv(filepath)
    else: 
        raise FileExistsError('[!] Dataset source file does not exists.')
    
def split_data(data: pd.DataFrame, label_column: str) -> Tuple[np.ndarray, np.ndarray]:
    y = data[label_column].to_numpy()
    x = data.drop(label_column, axis=1).to_numpy()
    return (x, y)

class MNIST:
    
    def __init__(self, test_split: float, load_test: bool=False) -> None:
        self.test_split = test_split
        self.test = load_test
        self.load_data()
        
    def load_data(self) -> None:
        if self.test:
            self.df_test = check_and_load(TEST_PATH)
        self.df_train = check_and_load(TRAIN_PATH)
        
    @property
    def dataset(self) -> np.ndarray:

        train_set = split_data(self.df_train, 'label')
        
        if self.test:
            test_set = split_data(self.df_test, 'label')
            return train_set, test_set
        
        X_train, X_test, y_train, y_test = train_test_split(*train_set,
                                                            test_size=self.test_split)
        return (X_train, y_train), (X_test, y_test)
    
    def show(self, subset: str, sample_id: int) -> None:
        if subset == 'train':
            data, label = split_data(self.df_train, 'label')
        else:
            if not hasattr(self, 'df_test'):
                raise AttributeError("Testing subset is not exist, please recreate " +
                                     "object with flag 'load_test'")
            data, label = split_data(self.df_test, 'label')
                
            
        plt.figure(figsize=(5,5))
        print(label[sample_id])
        plt.imshow(data[sample_id].reshape(28,28), cmap=plt.cm.gray)
            
            