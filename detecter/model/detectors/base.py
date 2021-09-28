# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from cvcore.cnn import BaseModule
import torch
from detecter.core.structures import ImageList

__all__ = ['BaseDetector']


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, comm_cfg, init_cfg=None, **kwargs):
        pixel_mean = comm_cfg['pixel_mean']
        pixel_std = comm_cfg['pixel_std']

        super(BaseDetector, self).__init__(init_cfg)
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @abstractmethod
    def extract_feat(self, imgs, batched_inputs=None):
        """Extract features from images."""
        pass

    def preprocess_image(self, batched_inputs):
        images = [x["img"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.extract_feat(images.tensor)
        return features
