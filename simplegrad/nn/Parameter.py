from simplegrad.Tensor import Tensor


class Parameter(Tensor):
    """Simple wrapper class arount simplegrad.Tensor to indicate a parameter. This is used to extract the trainable parameters of a network"""

    def __init__(self, params: Tensor):
        assert isinstance(params, Tensor), "parameter has to be a tensor"
        super().__init__(params)

        # automatically set required_grad True for all parameters
        # because if it's a parameter, we want to train it
        params.requires_grad = True
