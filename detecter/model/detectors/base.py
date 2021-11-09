# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from cvcore.cnn import BaseModule
import torch
from collections import OrderedDict
import torch.distributed as dist
from detecter.core.structures import ImageList

__all__ = ['BaseDetector']


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, comm_cfg, init_cfg=None, **kwargs):
        super(BaseDetector, self).__init__(init_cfg)

        pixel_mean = comm_cfg['pixel_mean']
        pixel_std = comm_cfg['pixel_std']
        self.to_rgb = comm_cfg['to_rgb']
        self.debug = comm_cfg['debug']

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
        if self.to_rgb and images[0].size(0) == 3:
            images = [image[[2, 1, 0], ...] for image in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.extract_feat(images.tensor)
        return features

    def parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
