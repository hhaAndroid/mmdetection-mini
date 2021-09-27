# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


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
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
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
        if self.training:
            losses = self.bbox_head(features, batched_inputs)
            return losses
        else:
            return self.bbox_head(features)
