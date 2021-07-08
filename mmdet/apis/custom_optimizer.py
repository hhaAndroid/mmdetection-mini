import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS


@OPTIMIZER_BUILDERS.register_module()
class CustomOptimizer:
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.optimizer_cfg = optimizer_cfg
        self.weight_decay = 0.0005

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
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
        optimizer_cfg['params'].append({'params': pg1, 'weight_decay': self.weight_decay})  # add pg1 with weight_decay
        optimizer_cfg['params'].append(({'params': pg2}))  # add pg2 (biases)

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
