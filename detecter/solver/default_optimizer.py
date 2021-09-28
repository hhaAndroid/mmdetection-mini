import copy

from .builder import TORCH_OPTIMIZERS, OPTIMIZERS
from .misc import filter_no_grad_params
from cvcore import build_from_cfg

__all__ = ['build_default_optimizer']


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
        return build_from_cfg(paramwise_cfg, TORCH_OPTIMIZERS, default_args)
