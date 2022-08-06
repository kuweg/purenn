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
    nodes = [layer.n_nodes for layer in model.layers]
    activations = _extract_activation_functions(model)
    weights_layers_shapes = _reversed_pairwise(nodes)
    weights_layers = [
            WeightsLayer(input_shape, output_shape, activation_fn)
            for (input_shape, output_shape), activation_fn
            in zip(weights_layers_shapes, activations)
        ]
    return WeigthCore(*weights_layers)
    

class Sequential(Model):
    
    def __init__(self, *args) -> None:
        self.layers = [layer for layer in args]
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