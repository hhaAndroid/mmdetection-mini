# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from cvcore import get_log_storage

__all__ = ['SingleStageDetector']


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 comm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(SingleStageDetector, self).__init__(comm_cfg, init_cfg, **kwargs)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img, batched_inputs=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img, batched_inputs)
        if self.with_neck:
            x = self.neck(x, batched_inputs)
        return x

    def forward(self, batched_inputs):
        features = super(SingleStageDetector, self).forward(batched_inputs)
        outputs = self.bbox_head(features, batched_inputs)
        if self.training:
            # outputs = self.bbox_head(features, batched_inputs)
            # TODO: remove
            loss, log_vars = self.parse_losses(outputs)
            log_storage = get_log_storage()
            log_storage.append(log_vars)
            return loss
        else:
            return outputs
