from cvcore import Registry, build_from_cfg

__all__ = ['DATASETS', 'PIPELINES', 'build_dataset']

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
