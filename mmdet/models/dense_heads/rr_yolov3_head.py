# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from .base_yolo_head import BaseYOLOHead
from ..utils import brick as vn_layer
from ..builder import HEADS


@HEADS.register_module()
class RRYolov3Head(BaseYOLOHead):

    def _init_layers(self):
        layer_list = [
            # stage 6, largest
            OrderedDict([
                ('1_head', vn_layer.Head(self.in_channels[0], self.num_anchors, self.num_classes)),
            ]),

            OrderedDict([
                ('2_head', vn_layer.Head(self.in_channels[1], self.num_anchors, self.num_classes)),
            ]),

            # smallest
            OrderedDict([
                ('3_head', vn_layer.Head(self.in_channels[2], self.num_anchors, self.num_classes)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, feats):
        stage6 = self.layers[0](feats[0])
        stage5 = self.layers[1](feats[1])
        stage4 = self.layers[2](feats[2])
        features = [stage6, stage5, stage4]
        return tuple(features),

