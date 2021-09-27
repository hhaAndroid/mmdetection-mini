# Copyright (c) OpenMMLab. All rights reserved.

from cvcore.utils import Registry, build_from_cfg

__all__ = ['PRIOR_GENERATORS', 'build_prior_generator']

PRIOR_GENERATORS = Registry('Generator for anchors and points')


def build_prior_generator(cfg, default_args=None):
    return build_from_cfg(cfg, PRIOR_GENERATORS, default_args)
