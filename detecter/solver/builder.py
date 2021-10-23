import copy
from cvcore import Registry, build_from_cfg, Hook
import torch
import inspect

__all__ = ['OPTIMIZERS', 'TORCH_OPTIMIZERS', 'LR_SCHEDULERS', 'PARAM_SCHEDULERS', 'build_optimizer',
           'build_lr_scheduler']

OPTIMIZERS = Registry('optimizer')
TORCH_OPTIMIZERS = Registry('torch_optimizer')
LR_SCHEDULERS = Registry('lr scheduler')
PARAM_SCHEDULERS = Registry('param scheduler')


def build_optimizer(cfg, model, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    cp_cfg['model'] = model
    optimizer = build_from_cfg(cp_cfg, OPTIMIZERS, default_args)
    return optimizer


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            TORCH_OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


register_torch_optimizers()



def build_lr_scheduler(cfg, optimizer, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    cp_cfg['optimizer'] = optimizer
    lr_scheduler = build_from_cfg(cp_cfg, LR_SCHEDULERS, default_args)
    return lr_scheduler
