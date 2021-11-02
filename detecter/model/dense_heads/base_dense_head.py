# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from cvcore.cnn import BaseModule
from cvcore import convert_image_to_rgb
from detecter.visualizer import get_event_storage
import torch
import copy

__all__ = ['BaseDenseHead']


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None, vis_interval=-1):
        super(BaseDenseHead, self).__init__(init_cfg)
        if isinstance(vis_interval, dict):
            self.train_vis_interval = vis_interval['train']
            self.val_vis_interval = vis_interval['val']
        else:
            self.train_vis_interval = vis_interval
            self.val_vis_interval = -1

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

            if self.val_vis_interval > 0:
                storage = get_event_storage()
                if storage.iter % self.val_vis_interval == 0:
                    with torch.no_grad():
                        imgs = []
                        data_samples = []
                        for (input, result) in zip(batched_inputs, results):
                            # result=result.new() # TODO
                            result = copy.deepcopy(result)  # TODO
                            imgs.append(input['img'])
                            scale_factor = input['img_metas']['scale_factor']
                            result.bboxes.scale(scale_factor[0], scale_factor[1])

                            data_sample = input["data_sample"]
                            data_sample.pred_instances = result.to('cpu')
                            data_samples.append(data_sample)

                        self._visualize_val(imgs, data_samples)
            return results

    def _visualize_val(self, images, data_samples):
        storage = get_event_storage()
        for (img, data_sample) in zip(images, data_samples):
            vis_img = convert_image_to_rgb(img.permute(1, 2, 0), "RGB")
            vis_name = "VAL:GT--Predicted"
            storage.add_image(vis_name, vis_img, data_sample)
            break  # only visualize one image in a batch
