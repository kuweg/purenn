from abc import ABC, abstractmethod
from prettytable import PrettyTable
from typing import Callable, List, Union
from tqdm import tqdm
import numpy as np

from .dataloader import DataLoader, _input_fit_handler
from .exceptions import IncompleteModelError
from .layers import Layer, WeightsLayer
from .utils import reversed_pairwise, add_vertical_row


WL_PREFIX = "wl{}"


class Model(ABC):
    
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, loss: np.float64) -> None:
        pass


class WeigthCore:
    
    def __init__(self, *args: List[Layer]):
        for order, layer in enumerate(args):
            setattr(self, WL_PREFIX.format(order), layer)
            
        self.n_layers = order + 1
        
    @property
    def layers(self):
        return [
                getattr(self, WL_PREFIX.format(layer_number))
                for layer_number in range(self.n_layers)
                ]
            
            
def check_completeness(model) -> bool:
        return all(
            list(model.__dict__.values())[:-1]
            )
    
    
def get_none_parameters(model) -> list[str]:
    return [k for k, v in model.__dict__.items() if v is None]


def _extract_activation_functions(model: Model) -> list[Callable]:
    activations = [layer.activation for layer in model.layers]
    return activations

def _extract_weights_strategies(model: Model) -> list[str]:
    ws = [layer.weights_strategy for layer in model.layers]
    return ws


def _weights_layers_init(model: Model) -> list[tuple]:
    nodes = [model.input_shape[1]] + [layer.n_nodes for layer in model.layers]
    activations = _extract_activation_functions(model)
    w_strategies = _extract_weights_strategies(model)
    weights_layers_shapes = reversed_pairwise(nodes)
    weights_layers = [
            WeightsLayer(input_shape, output_shape, activation_fn, ws)
            for (input_shape, output_shape), activation_fn, ws
            in zip(weights_layers_shapes, activations, w_strategies)
        ]
    return WeigthCore(*weights_layers)            

class Sequential(Model):
    
    def __init__(self,
                 input_shape: tuple,
                 layers: List[Layer],
                 optimizer: Callable=None,
                 loss: Callable=None) -> None:
        self.input_shape = input_shape
        self.layers = layers
        self.weights = _weights_layers_init(self)
        self.optimzier = optimizer
        self.loss = loss
        self.stat = {}
        
    def compile(self, optimizer: Callable, loss: Callable) -> None:
        self.optimzier = optimizer
        self.loss = loss
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data.copy()
        layers = self.weights.layers
        for layer_number in range(len(self.layers)):
            layer = layers[layer_number]
            output = layer(output)
        return output
    
    
    def backward(self, loss: np.float64) -> None:
        for layer in reversed(self.weights.layers):
            loss, dw, db = layer.backward(loss)
            self.optimzier.update(layer, dw, db)
    
    def fit(self,
            X_train: Union[np.ndarray, DataLoader],
            y_train: np.ndarray=None,
            batch_size: int=1,
            epochs: int=0) -> None:
        
        self.completeness_handler()
        data = _input_fit_handler(X_train, y_train)
            
        self.stat['losses'] = []
        print('Start training for {} epochs'.format(epochs))
        for e in range(epochs):
            epoch_loss = []
            with tqdm(data, unit='samples') as tepoch:
                for x_i, y_i in tepoch:
                    tepoch.set_description('Epoch {}'. format(e+1))
                    y_hat = self.forward(x_i)
                    stat_loss = self.loss(y_i, y_hat) / batch_size
                    epoch_loss.append(stat_loss.flatten())
                    
                    loss = self.loss.df(y_i, y_hat)
                    
                    self.backward(loss)
                    self.stat['losses'].append(epoch_loss)
                
                print('{}: {}'.format(self.loss, stat_loss))
                    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.forward(input_data).flatten()
    
    def completeness_handler(self) -> None:
        if  not check_completeness(self):
            raise IncompleteModelError(
                'Model missing {} attributes'.format(
                    get_none_parameters(self)
                )
            )
            
    def info(self) -> None:
        desc = PrettyTable()
        model_name = 'model: {}'.format(self.__class__.__name__)
        desc.title = model_name
        desc.field_names = ['Layer type',
                            'Weights shape',
                            'Bias shape',
                            'W strategy',
                            'Activation']
        layers_type = [layer.__class__.__name__ for layer in self.layers]
        layers_info = [
                        layer_type + '|' + str(layer)
                        for layer, layer_type
                        in zip(
                            list(self.weights.__dict__.values()),
                            layers_type
                            )
                        ]
        for layer_info in layers_info:
            desc.add_row(layer_info.split('|'))
        print(desc)
        opt, b_opt = add_vertical_row(desc,
                                      'Optimizer',
                                      str(self.optimzier))
        loss, b_loss = add_vertical_row(desc,
                                        'Loss',
                                        str(self.loss))
        print(opt)
        print(b_opt)
        print(loss)
        print(b_loss)
        
