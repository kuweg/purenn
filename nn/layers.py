from .activations import relu
from .weights import Weights, init_bias

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Callable, Union


class Layer(ABC):
    
    @abstractmethod
    def __call__(self, input_data):
        pass
    
class ConsctructionLayer(ABC):
    """Creating a structure of neural network at Model object."""
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes


class InputLayer(ConsctructionLayer):
    pass    


NO_ACTIVATION_LAYERS_LIST = [InputLayer]


class Dense(ConsctructionLayer):
    
    def __init__(self, n_nodes: int, activation: Callable) -> None:
        super().__init__(n_nodes)
        self.activation = activation

class WeightsLayer(Layer):
    
    def __init__(self,
                 n_input_nodes: int,
                 n_output_nodes: int,
                 activation: Callable,
                 weights_strategy: str = 'rand') -> None:
        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        self.activation = activation
        self._weights_init = Weights(weights_strategy)
        self.weights = self._weights_init(n_input_nodes, n_output_nodes)
        self.bias = np.random.rand(n_input_nodes)
        self.input = None
        self.output = None
        
    def forward(self, input_data: Union[Layer, np.ndarray]) -> np.ndarray:
         
        if isinstance(input_data, Layer):
             input_data = input_data.weights
             
        self.input = input_data
        self.z = np.dot(self.weights, input_data) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, error: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    
        input_error = np.dot(self.weights.T, error)
        
        dz = self.activation(input_error, derivative=True)
        
        weights_error = np.dot(self.input.T, dz)
        self.weights -= (alpha * weights_error)
        
        return dz
    
    def __repr__(self):
        return "({},{}) : {}".format(self.n_input_nodes, self.n_output_nodes, self.activation)
    
    __call__ = forward
    
    
        
        
     
