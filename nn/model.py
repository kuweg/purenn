from abc import ABC, abstractmethod

from itertools import pairwise
from typing import Callable, Iterable, List
import numpy as np

from nn.layers import Layer, WeightsLayer, NO_ACTIVATION_LAYERS_LIST


WL_PREFIX = "wl{}"


class Model(ABC):
    
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass


class WeigthCore:
    
    def __init__(self, *args: List[Layer]):
        for order, layer in enumerate(args):
            setattr(self, WL_PREFIX.format(order), layer)
        setattr(self, '_len', len(args))
        


def _reversed_pairwise(sequence: Iterable) -> list[list]:
    pairwise_sequence = list(pairwise(sequence))
    reversed_pairwise_sequence = list(
        map(
            lambda pair: list(reversed(pair)),
            pairwise_sequence
            )
        )
    return reversed_pairwise_sequence


def _extract_activation_functions(model: Model) -> list[Callable]:
    activations = [layer.activation for layer in model.layers
                   if type(layer) not in NO_ACTIVATION_LAYERS_LIST]
    return activations


def _weights_layers_init(model: Model) -> list[tuple]:
    nodes = [model.input_shape[1]] + [layer.n_nodes for layer in model.layers]
    activations = _extract_activation_functions(model)
    weights_layers_shapes = _reversed_pairwise(nodes)
    weights_layers = [
            WeightsLayer(input_shape, output_shape, activation_fn)
            for (input_shape, output_shape), activation_fn
            in zip(weights_layers_shapes, activations)
        ]
    return WeigthCore(*weights_layers)
    

class Sequential(Model):
    
    def __init__(self,
                 input_shape: tuple,
                 layers: List[Layer]) -> None:
        self.input_shape = input_shape
        self.layers = [layer for layer in layers]
        self.weights = _weights_layers_init(self)
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data.copy()
        for layer_number in range(self.weights._len):
            layer = getattr(self.weights, WL_PREFIX.format(layer_number))
            output = layer(output)
        return output
    
    def back_propogation(self):
        pass
    
    def fit(self, X_train, Y_train, epochs):
        pass
    
    def info(self) -> None:
        layers_type = [layer.__class__.__name__ for layer in self.layers]
        print('model: Sequential')
        print(
            "\n".join(
            [
                layer_type + " | " + str(layer) 
                for layer, layer_type
                in zip(
                    list(self.weights.__dict__.values())[:-1],
                    layers_type
                    )
                ]
            )
        )