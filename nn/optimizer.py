from abc import ABC, abstractmethod
from .layers import Layer


class Optimizer(ABC):
    
    @abstractmethod
    def update(self,
               layer: Layer,
               weight_error: float,
               bias_erorr: float) -> None:
        pass
    
    def __repr__(self) -> str:
        return self.__class__.__name__

    
class GradientDescent(Optimizer):
    
    def __init__(self, lr: float=0.01) -> None:
        self.lr = lr
        
    def update(self,
               layer: Layer,
               weight_error: float,
               bias_error: float) -> None:

        layer.update_params(
            dw = weight_error * self.lr,
            db = bias_error * self.lr
        )


class SGD(Optimizer):
    
    def __init__(self,
                 lr: float=0.01,
                 momentum: float=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.m_update = 0
        
    def update(self,
               layer: Layer,
               weight_error: float,
               bias_error: float) -> None:
        self.m_update = -self.lr*weight_error
        layer.update_params(
            dw = weight_error * self.lr,
            db = bias_error * self.lr
        )