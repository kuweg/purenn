from .weights import Weights
from .activations import set_activation
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union


class Layer(ABC):
    
    @abstractmethod
    def forward(self, input_data):
        pass
    
    @abstractmethod
    def backward(self, input_data):
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
                 activation: Union[Callable, str],
                 weights_strategy: str = 'rand') -> None:

        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        self.activation = set_activation(activation)
        self._weights_init = Weights(weights_strategy)
        self.weights = self._weights_init(n_input_nodes, n_output_nodes)
        self.bias = np.random.rand(n_input_nodes, 1)
        
    def forward(self, input_data: Union[Layer, np.ndarray]) -> np.ndarray:     
        self.input = input_data
        self.z = np.dot(self.weights, input_data) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, error: np.ndarray) -> np.ndarray:
        batch_size = self.input.shape[1]
        dz = self.activation(self.z, derivative=True) * error
            
        dw = (1 / batch_size) * np.dot(dz, self.input.T)
        db = (1 / batch_size) * np.sum(dz, axis=1, keepdims=True)
    
        new_error = np.dot(self.weights.T, dz)
        return new_error, dw, db
    
    def update_params(self, dw: np.ndarray, db: np.ndarray) -> None:
        self.weights -= dw
        self.bias -= db
        
    
    def __repr__(self):
        return "({},{}) <{}> | {}".format(
            self.n_input_nodes,
            self.n_output_nodes,
            self.bias.shape,
            self.activation
            )
    
    __call__ = forward
    
    
        
        
     
