import csv
import gzip
from pathlib import Path
import tempfile
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import os

import requests

from nn.utils import flatten_list

MNIST_SMALL_FILE = 'mnist.csv'
MNIST_BIG_FILE = 'mnist_big.csv'

MNIST_ARCHIVE_SMALL = ['mnist_data.gz', 'mnist_labels.gz']
MNIST_ARCHIVE_BIG = ['mnist_data_b.gz', 'mnist_labels_b.gz']

DATA_PATH = Path('datasets')

MNIST_SMALL_PATH = DATA_PATH / MNIST_SMALL_FILE
MNIST_BIG_PATH = DATA_PATH / MNIST_BIG_FILE

URLS = {
        'big_data': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'big_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'data': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

ARCHIVE_TO_CSV_PATH_MAPPING = {
    (MNIST_ARCHIVE_SMALL[0], MNIST_ARCHIVE_SMALL[1]) : MNIST_SMALL_PATH,
    (MNIST_ARCHIVE_BIG[0], MNIST_ARCHIVE_BIG[1]) : MNIST_BIG_PATH
}

URL_TO_ARCHIVE_MAPPING = {
        URLS['big_data']: MNIST_ARCHIVE_BIG[0],
        URLS['big_labels']: MNIST_ARCHIVE_BIG[1],
        URLS['data']: MNIST_ARCHIVE_SMALL[0],
        URLS['labels']: MNIST_ARCHIVE_SMALL[1]
    }

URL_TO_CSV_MAPPING = {
    (URLS['data'], URLS['labels']): MNIST_SMALL_FILE,
    (URLS['big_data'], URLS['big_labels']): MNIST_BIG_FILE
}

MODE_TO_URL_MAPPING = {
    'small': [URLS['data']] + [URLS['labels']], 
    'big': [URLS['big_data']] + [URLS['big_labels']],
    'full': list(URLS.values())
}

    
def split_data(data: pd.DataFrame, label_column: str) -> Tuple[np.ndarray, np.ndarray]:
    y = data[label_column].to_numpy()
    x = data.drop(label_column, axis=1).to_numpy()
    return (x, y)


def map_url_to_filename(urls: List[str], mapping_dict: dict) -> list:
    return list(
                map(
                    lambda sample: mapping_dict[sample],
                    urls
                    )
                )


def download_url(url: str, save_path: str, chunk_size=1024) -> None:
        
        try:
            file = requests.get(url, stream=True)
            
            if file.status_code != 200:
                raise FileExistsError('Could not download file, check link')
        
            with open(save_path, 'wb') as f:
                for chunk in file.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                    
        except Exception as ex:
            raise ex
                
            
def unpzip_mnist_images(file_path: str):
    with gzip.open(file_path, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images


def unzip_mnist_labels(file_path: str):
    with gzip.open(file_path, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels
    
    
def write_mnist_to_scv(save_path: str, data: Tuple[np.ndarray]) -> None:
    """
    Write MNIST data into .csv file
    
    :param save_path: a path where file will be saved
    :type save_path: str
    :param data: A tuple of (x, y)
    :type data: tuple
    """
    header = ['target'] + ['pixel{}'.format(order) for order in range(784)]
    with open(save_path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for sample, label in zip(*data):
            row = np.insert(sample.flatten(), 0, label)
            writer.writerow(row)
            
            
def check_files_existance(urls_list: List[str]) -> Union[List[str], None]:
    """
    Checking that required files are exist. If they aren't
    returns list of missing files.
    """
    needed_csv_files = [
                URL_TO_CSV_MAPPING[(data_url, label_url)]
                for data_url, label_url in
                zip(urls_list[::2], urls_list[1::2])
                ]
    dir_list = os.listdir(DATA_PATH)    
    files_intersec = set(dir_list) & set(needed_csv_files)
        
    if files_intersec == set(needed_csv_files):
        print('All required files are exist!')
        return None
        
    files_to_download = files_intersec ^ set(needed_csv_files)
    print('Missing {} file(s). Downloading'.format(files_to_download))
    return files_to_download
    
def get_missing_urls(required_files: List[str]) -> list[str]:
    """
    Converting list of required (missing) files into corresponding
    urls.
    """
    reversed_url_to_csv = {value: key for key, value in URL_TO_CSV_MAPPING.items()}
    required_urls = [list(reversed_url_to_csv[file]) for file in required_files]
    required_urls = flatten_list(required_urls)
    return required_urls


class MNISTModeException(Exception):
    pass

class MNIST:
    """
    Class for getting Mnist dataset. Loads .csv file(s) if not exist(s).
    Otherwise use exisiting file(s).
    Constructs dataframe(s) with (n_samples, 785) shape,
    where first column is a target label and rest 784 are
    flattened (28, 28) image.
    Dataframe(s) can be called using .df (.df_train, .df_test)
    after object initialization.
    
    :param mode: Use one of the ['small', 'big', 'full] flags
                 for getting a dataset
    :type mode: str
    :return: small -> (x, y)
             big -> (x, y)
             full -> (x_train, y_train), (x_test, y_test)
    :rtype: Union[Tuple, Tuple[Tuple]]
    """
 
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.setup_data()
                    
    def process_urls(self) -> List[str]:
        """
        Mapping given mode with mapping dict.
        As a result gives a list of archives urls, which will be downloaded.
        """
        try:
            files_to_load = MODE_TO_URL_MAPPING[self.mode]
        except KeyError as exc:
            raise MNISTModeException(
                "Unknow parameter for mode, use one of these -> ['small', 'big', 'full']"
                ) from None
        
        return files_to_load
    
    
    def _get_files_to_load(self) -> Union[List[str], None]:
        """
        Forming list of missing files to download.
        Otherwise return None.
        """
        files_to_load = self.process_urls()
        missing_files = check_files_existance(files_to_load)
        
        if not missing_files:
            return None
        
        files_to_load = get_missing_urls(missing_files)   
        
        return files_to_load

    def download_data(self) -> None:
        """
        Downloading Yann LeCun's MNIST dataset.
        Stores archives into temporary directory,
        reads binary files and puts them into csv file(s).
        """
        files_to_load = self._get_files_to_load()
        
        if not files_to_load:
            return None
        else:
            
            archive_names = map_url_to_filename(files_to_load, URL_TO_ARCHIVE_MAPPING)
            print(files_to_load)
            print(archive_names)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                file_pathes = [temp_path / file_name 
                            for file_name in archive_names]
                
                for file_url, file_path in zip(files_to_load, file_pathes):
                    download_url(file_url, file_path)
                    
                for data_path, label_path in zip(file_pathes[::2], file_pathes[1::2]):
                    
                    x_targets = unpzip_mnist_images(data_path)
                    y_labels = unzip_mnist_labels(label_path)
                    
                    csv_file_path = ARCHIVE_TO_CSV_PATH_MAPPING[(data_path.name, label_path.name)]
                    
                    write_mnist_to_scv(csv_file_path, (x_targets, y_labels))
                    
    def setup_data(self) -> None:
        
        self.download_data()
        
        if self.mode == 'small':
            self.df = pd.read_csv(MNIST_SMALL_PATH)
        elif self.mode == 'big':
            self.df = pd.read_csv(MNIST_BIG_PATH)
        elif self.mode == 'full':
            self.df_train = pd.read_csv(MNIST_BIG_PATH)
            self.df_test = pd.read_csv(MNIST_SMALL_PATH)
        

        
    @property
    def dataset(self) -> np.ndarray:
        
        if self.mode == 'full':
            train_set = split_data(self.df_train, 'target')
            test_set = split_data(self.df_test, 'target')
            return train_set, test_set

        train_set = split_data(self.df, 'target')
        
        return train_set
            