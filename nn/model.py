from abc import ABC, abstractmethod

from itertools import pairwise
from typing import Callable, Iterable, List
import numpy as np

from nn.layers import Layer, WeightsLayer
from nn.utils import reversed_pairwise


WL_PREFIX = "wl{}"


class Model(ABC):
    
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass


class WeigthCore:
    
    def __init__(self, *args: List[Layer]):
        for order, layer in enumerate(args):
            setattr(self, WL_PREFIX.format(order), layer)
        
    
class Sequential(Model):
    
    def __init__(self,
                 input_shape: tuple,
                 layers: List[Layer],
                 loss: Callable) -> None:
        self.input_shape = input_shape
        self.layers = [layer for layer in layers]
        self.weights = _weights_layers_init(self)
        self.loss = loss
        self.stat = {}
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data.copy()
        for layer_number in range(len(self.layers)):
            layer = getattr(self.weights, WL_PREFIX.format(layer_number))
            output = layer(output)
        return output
    
    
    def back_propogation(self):
        pass
    
    def fit(self, X_train, Y_train, epochs, alpha: float=0.1):
        
        self.stat['losses'] = []
        wl = list(self.weights.__dict__.values())
        print('Start training for {} epochs'.format(epochs))
        for e in range(epochs):
            epoch_loss = []
            for x_i, y_i in zip(X_train, Y_train):
                
                y_hat = self.forward(x_i)
                stat_loss = self.loss(y_i, y_hat)
                epoch_loss.append(stat_loss.flatten())
                
                loss = self.loss.df(y_i, y_hat)
                
                for layer in reversed(wl):
                    loss, dw, db = layer.backward(loss)
                    layer.update_params(dw=dw*alpha, db=db*alpha)
            self.stat['losses'].append(epoch_loss)
            
            print('Epoch : {} - Loss: {}'.format(e, stat_loss))
                    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.forward(input_data).flatten()
        
    
    def info(self) -> None:
        layers_type = [layer.__class__.__name__ for layer in self.layers]
        print('model: {}'.format(self.__class__.__name__))
        print(
            "\n".join(
            [
                layer_type + " | " + str(layer) 
                for layer, layer_type
                in zip(
                    list(self.weights.__dict__.values()),
                    layers_type
                    )
                ]
            )
        )
        
        
def _extract_activation_functions(model: Model) -> list[Callable]:
    activations = [layer.activation for layer in model.layers]
    return activations


def _weights_layers_init(model: Model) -> list[tuple]:
    nodes = [model.input_shape[1]] + [layer.n_nodes for layer in model.layers]
    activations = _extract_activation_functions(model)
    weights_layers_shapes = reversed_pairwise(nodes)
    weights_layers = [
            WeightsLayer(input_shape, output_shape, activation_fn)
            for (input_shape, output_shape), activation_fn
            in zip(weights_layers_shapes, activations)
        ]
    return WeigthCore(*weights_layers)