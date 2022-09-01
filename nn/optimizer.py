from .layers import Layer


class GradientDescent:
    
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

    def __repr__(self) -> str:
        return self.__class__.__name__