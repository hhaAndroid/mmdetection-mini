# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from cvcore.cnn import BaseModule

__all__ = ['BaseDenseHead']


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None, vis_interval=-1):
        super(BaseDenseHead, self).__init__(init_cfg)
        self.vis_interval = vis_interval

    def _init_layers(self):
        """Initialize layers of the head."""
        raise NotImplementedError

    def forward_layer(self, features, batched_inputs=None):
        raise NotImplementedError

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        pass

    def forward(self, features, batched_inputs=None, **kwargs):
        output = self.forward_layer(features, batched_inputs)
        if self.training:
            loss = self.loss(*output, batched_inputs=batched_inputs, **kwargs)
            return loss
        else:
            results = self.get_bboxes(*output, batched_inputs=batched_inputs, **kwargs)
            return results
