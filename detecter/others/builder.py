from cvcore import Registry, build_from_cfg

__all__ = ['FUNCSTORAGES', 'build_func_storage']

FUNCSTORAGES = Registry('func_storages')


def build_func_storage(cfg):
    return build_from_cfg(cfg, FUNCSTORAGES)


