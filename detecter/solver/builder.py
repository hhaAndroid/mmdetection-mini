import copy
from cvcore import Registry, build_from_cfg, Hook
import torch
import inspect

__all__ = ['OPTIMIZERS', 'TORCH_OPTIMIZERS', 'LR_SCHEDULERS', 'build_optimizer', 'build_lr_scheduler']

OPTIMIZERS = Registry('optimizer')
LR_SCHEDULERS = Registry('lr_scheduler')


def build_optimizer(cfg, model, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    cp_cfg['model'] = model
    optimizer = build_from_cfg(cp_cfg, OPTIMIZERS, default_args)
    return optimizer


TORCH_OPTIMIZERS = Registry('torch_optimizer')


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


# 如果内部不采用 hook 模式，那么感觉通用性和可复用性很难保证
# 故内部依然采用 hook 实现，但是对外不暴露
def build_lr_scheduler(cfg, optimizer, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    cp_cfg['optimizer'] = optimizer
    lr_scheduler = build_from_cfg(cp_cfg, LR_SCHEDULERS, default_args)
    assert isinstance(lr_scheduler, Hook)
    return lr_scheduler
