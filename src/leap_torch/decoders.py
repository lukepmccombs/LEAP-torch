from leap_ec import Decoder
import numpy as np
import torch
from torch import nn


class NumpyWrapper:
    """
    Wraps a module instance to convert it's inputs and output to a numpy array.
    Only supports modules whose inputs are convertable to tensors and
    whose return value is a single tensor.
    """
    
    def __init__(self, module, dtype=np.float64, device="cpu", nograd=True):
        """
        :param module: the pytorch module wrapped by this class.
        :param dtype: the dtype the return tensor of the module will be
            cast to. Defaults to np.float64.
        :param nograd: whether pytorch should be run in nograd mode when
            the module is called. May accelerate computation. Defaults
            to True.
        """
        self.module = module.to(device)
        self.dtype = dtype
        self.device = device
        self.nograd = nograd
    
    def __call__(self, *args, **kwargs):
        # Convert args to tensors on device
        args = [torch.tensor(a, device=self.device) for a in args]
        kwargs = {k: torch.tensor(a, device=self.device) for k, a in kwargs.items()}
        
        if self.nograd:
            with torch.no_grad():
                ret = self.module(*args, **kwargs)
        else:
            ret = self.module(*args, **kwargs)
            
        # Detaches the tensor from its graph, brings it over to
        # the cpu, and then converts it to numpy.
        return ret.detach().cpu().numpy().astype(self.dtype)


class NumpyDecoder(Decoder):
    """
    Decodes modules by wrapping them with a class that automatically
    converts inputs to tensors and return values to numpy arrays.
    Only supports modules whose inputs are convertable to tensors and
    whose return value is a single tensor.
    """
    
    def __init__(self, dtype=np.float64, device=None, nograd=True):
        """
        :param dtype: the dtype the return tensor of the module will be
            cast to. Defaults to np.float64
        :param device: the device the module and arguments will be placed on.
        :param nograd: whether pytorch should be run in nograd mode when
            the module is called. May accelerate computation. Defaults
            to True.
        """
        self.dtype = dtype
        self.device = device
        self.nograd = nograd
    
    def decode(self, genome: nn.Module, *_, **__):
        """
        :param genome: the pytorch module to be wrapped.
        :return: a NumpyWrapper instance wrapping the module.
        """
        assert isinstance(genome, nn.Module),\
            "Wrapped genome must be a pytorch Module"
        return NumpyWrapper(genome, self.dtype, self.device, self.nograd)
