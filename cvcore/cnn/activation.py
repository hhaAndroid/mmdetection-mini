# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

__all__ = ['build_activation_layer', 'ACTIVATION_LAYERS']

from mmcv.utils import TORCH_VERSION, build_from_cfg, digit_version
from .registry import ACTIVATION_LAYERS

for module in [
    nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
    nn.Sigmoid, nn.Tanh
]:
    ACTIVATION_LAYERS.register_module(module=module)

if (TORCH_VERSION == 'parrots'
        or digit_version(TORCH_VERSION) < digit_version('1.4')):
    ACTIVATION_LAYERS.register_module(module=GELU)
else:
    ACTIVATION_LAYERS.register_module(module=nn.GELU)


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)
