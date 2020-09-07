from functools import partial

import torch

TORCH_VERSION = torch.__version__


def _get_cuda_home():
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


def get_build_config():
    return torch.__config__.show()


def _get_conv():
    from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin


def _get_dataloader():
    from torch.utils.data import DataLoader
    PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def _get_extension():
    from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                               CUDAExtension)
    return BuildExtension, CppExtension, CUDAExtension


def _get_pool():
    from torch.nn.modules.pooling import (_AdaptiveAvgPoolNd,
                                              _AdaptiveMaxPoolNd, _AvgPoolNd,
                                              _MaxPoolNd)
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd


def _get_norm():
    from torch.nn.modules.instancenorm import _InstanceNorm
    from torch.nn.modules.batchnorm import _BatchNorm
    return _BatchNorm, _InstanceNorm


CUDA_HOME = _get_cuda_home()
_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
BuildExtension, CppExtension, CUDAExtension = _get_extension()
_BatchNorm, _InstanceNorm = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()

