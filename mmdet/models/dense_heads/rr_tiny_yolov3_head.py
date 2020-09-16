# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
import torch
from .base_yolo_head import BaseYOLOHead
from ..utils import brick as vn_layer
from ..builder import HEADS


@HEADS.register_module()
class RRTinyYolov3Head(BaseYOLOHead):

    def _init_layers(self):
        layer_list = [
            # stage 6
            OrderedDict([
                ('14_convbatch', vn_layer.Conv2dBatchLeaky(self.in_channels[0], self.out_channels[0], 3, 1)),
                ('15_conv', nn.Conv2d(self.out_channels[0], self.num_anchors * self.num_attrib, 1, 1, 0)),
            ]),
            # stage 5
            # stage5 / upsample
            OrderedDict([
                ('18_convbatch', vn_layer.Conv2dBatchLeaky(self.in_channels[1], self.out_channels[1], 1, 1)),
                ('19_upsample', nn.Upsample(scale_factor=2)),
            ]),
            # stage5 / head
            OrderedDict([
                ('21_convbatch', vn_layer.Conv2dBatchLeaky(self.in_channels[1] + self.out_channels[1], self.out_channels[1]*2, 3, 1)),
                ('22_conv', nn.Conv2d(self.out_channels[1]*2, self.num_anchors * self.num_attrib, 1, 1, 0)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, feats):
        stage6 = self.layers[0](feats[0])
        stage5_upsample = self.layers[1](feats[0])
        stage5 = self.layers[2](torch.cat((stage5_upsample, feats[1]), 1))
        features = [stage6, stage5]
        return tuple(features),

