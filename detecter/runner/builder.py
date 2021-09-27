# Copyright (c) OpenMMLab. All rights reserved.

from cvcore import Registry, build_from_cfg

__all__ = ['RUNNERS', 'build_runner']

RUNNERS = Registry('runner')

def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, RUNNERS, default_args)
