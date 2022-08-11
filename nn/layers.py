from .weights import Weights

from abc import ABC, abstractmethod
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
        self.bias = np.random.rand(n_input_nodes, 1)
        
    def forward(self, input_data: Union[Layer, np.ndarray]) -> np.ndarray:
         
        if isinstance(input_data, Layer):
             input_data = input_data.weights
             
        self.input = input_data
        self.z = np.dot(self.weights, input_data) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, error: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    
        dz = self.activation(self.z, derivative=True) * error
            
        dw = np.dot(dz, self.input.T)
        db = np.sum(dz, axis=1, keepdims=True)
    
        new_error = np.dot(self.weights.T, dz)
        return new_error, dw, db
    
    def update_params(self, dw: np.ndarray, db: np.ndarray) -> None:
        self.weights -= dw
        self.bias -= db
        
    
    def __repr__(self):
        return "({},{}) | {}".format(self.n_input_nodes, self.n_output_nodes, self.activation)
    
    __call__ = forward
    
    
        
        
     
