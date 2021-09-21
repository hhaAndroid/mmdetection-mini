from .builder import OPTIMIZERS
from .misc import filter_no_grad_params
from cvcore import build_from_cfg


__all__=['build_default_optimizer']


@OPTIMIZERS.register_module()
def build_default_optimizer(cfg, model, default_args=None):
    assert 'optimizer_cfg' in cfg

    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = cfg.optimizer_cfg.copy()

    # if no paramwise option is specified, just use the global setting
    if 'paramwise_cfg' not in cfg or cfg['paramwise_cfg'] is None:
        optimizer_cfg['params'] = filter_no_grad_params(model, optimizer_cfg)
        return build_from_cfg(optimizer_cfg, OPTIMIZERS, default_args)
    else:
        # TODO  
        cp_cfg = cfg.pop('paramwise_cfg')
        return build_from_cfg(cp_cfg, OPTIMIZERS, default_args)
