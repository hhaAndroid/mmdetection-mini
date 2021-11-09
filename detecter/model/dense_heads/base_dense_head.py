# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from cvcore.cnn import BaseModule
from detecter.others import get_func_storage

__all__ = ['BaseDenseHead']


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, debug=False, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)
        self.debug = debug

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

            if self.debug:
                get_func_storage().visualize_val(dict(results=results, batched_inputs=batched_inputs))

            return results
