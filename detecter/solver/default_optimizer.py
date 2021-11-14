import copy

from .builder import TORCH_OPTIMIZERS, OPTIMIZERS
from .misc import filter_no_grad_params
from cvcore import build_from_cfg
import torch.nn as nn


__all__ = ['build_default_optimizer', 'build_yolov5_optimizer']


@OPTIMIZERS.register_module()
def build_default_optimizer(optimizer_cfg, paramwise_cfg, model, default_args=None):
    if hasattr(model, 'module'):
        model = model.module

    cp_optimizer_cfg = copy.deepcopy(optimizer_cfg)
    # if no paramwise option is specified, just use the global setting
    if paramwise_cfg is None:
        cp_optimizer_cfg['params'] = filter_no_grad_params(model, cp_optimizer_cfg)
        return build_from_cfg(cp_optimizer_cfg, TORCH_OPTIMIZERS, default_args)
    else:
        # TODO
        return build_from_cfg(optimizer_cfg, TORCH_OPTIMIZERS, default_args)



@OPTIMIZERS.register_module()
def build_yolov5_optimizer(optimizer_cfg, weight_decay, model, default_args=None):
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    optimizer_cfg['params'] = []

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer_cfg['params'].append({'params': pg0})
    optimizer_cfg['params'].append({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optimizer_cfg['params'].append(({'params': pg2}))  # add pg2 (biases)

    return build_from_cfg(optimizer_cfg, TORCH_OPTIMIZERS, default_args)
