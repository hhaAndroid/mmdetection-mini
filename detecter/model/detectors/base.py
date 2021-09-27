# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from cvcore.cnn import BaseModule

__all__ = ['BaseDetector']


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @abstractmethod
    def extract_feat(self, imgs, batched_inputs=None):
        """Extract features from images."""
        pass

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.extract_feat(images)
        return features