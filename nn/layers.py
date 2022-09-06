from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

from .weights import Weights
from .activations import set_activation, is_softmax
from .utils import dummy_callable


class Layer(ABC):
    
    @abstractmethod
    def forward(self, input_data) -> None:
        pass
    
    @abstractmethod
    def backward(self, input_data) -> None:
        pass
    
    @abstractmethod
    def update_params(self, dw: np.ndarray, db: np.ndarray) -> None:
        pass
    

class ConsctructionLayer(ABC):
    """Creating a structure of neural network at Model object."""
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

class Dense(ConsctructionLayer):
    
    def __init__(self,
                 n_nodes: int,
                 activation: Callable=dummy_callable,
                 weights_strategy: str='xavier') -> None:
        super().__init__(n_nodes)
        self.activation = activation
        self.weights_strategy = weights_strategy

class WeightsLayer(Layer):
    
    def __init__(self,
                 n_input_nodes: int,
                 n_output_nodes: int,
                 activation: Union[Callable, str]=dummy_callable,
                 weights_strategy: str=None) -> None:

        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        self.activation = set_activation(activation)
        self._weights_init = Weights(weights_strategy)
        self.weights = self._weights_init(n_input_nodes, n_output_nodes)
        self.bias = np.random.randn(n_input_nodes, 1)
        
    def forward(self, input_data: Union[Layer, np.ndarray]) -> np.ndarray:     
        self.input = input_data
        self.z = np.dot(self.weights, input_data) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, error: np.ndarray) -> np.ndarray:
        batch_size = self.input.shape[1]
        if is_softmax(self.activation):
            y = error * (-self.a)
            dz = self.a - y
        else:
            dz = self.activation(self.z, derivative=True) * error
            
        dw = (1 / batch_size) * np.dot(dz, self.input.T)
        db = (1 / batch_size) * np.sum(dz, axis=1, keepdims=True)
    
        new_error = np.dot(self.weights.T, dz)
        return new_error, dw, db
    
    def update_params(self, dw: np.ndarray, db: np.ndarray) -> None:
        self.weights -= dw
        self.bias -= db
        
    
    def __repr__(self) -> str:
        return "({}, {})|{}|{}|{}".format(
            self.n_input_nodes,
            self.n_output_nodes,
            self.bias.shape,
            self._weights_init.weights_strategy,
            self.activation
            )
    
    __call__ = forward
    
    
        
        
     
