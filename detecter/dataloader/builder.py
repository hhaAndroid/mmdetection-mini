import copy
from cvcore import Registry, build_from_cfg

__all__ = ['SAMPLER', 'DATALOADER', 'build_dataloader']

SAMPLER = Registry('sampler')
DATALOADER = Registry('dataloader')


def build_dataloader(cfg, dataset, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    cp_cfg['dataset'] = dataset
    dataloader = build_from_cfg(cp_cfg, DATALOADER, default_args)
    return dataloader

