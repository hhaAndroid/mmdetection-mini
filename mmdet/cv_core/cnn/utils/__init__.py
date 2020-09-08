# Copyright (c) Open-MMLab. All rights reserved.
from .weight_init import (bias_init_with_prob,
                          constant_init, kaiming_init, normal_init,
                          uniform_init, xavier_init)

__all__ = [
 'bias_init_with_prob',
    'constant_init', 'kaiming_init', 'normal_init', 'uniform_init',
    'xavier_init',
]
