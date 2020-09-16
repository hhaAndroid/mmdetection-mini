# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn as nn
from ..utils import brick as vn_layer
from ..builder import NECKS

__all__ = ['RRDarkNeck', 'RRYoloV4DarkNeck']


# FPN结构
@NECKS.register_module()
class RRDarkNeck(nn.Module):
    def __init__(self, input_channels=32):
        super(RRDarkNeck, self).__init__()
        layer_list = [
            # the following is extra
            # layer 3
            # output third scale, largest
            OrderedDict([
                ('head_body_1', vn_layer.HeadBody(input_channels * (2 ** 5), first_head=True)),
            ]),

            # layer 4
            OrderedDict([
                ('trans_1', vn_layer.Transition(input_channels * (2 ** 4))),
            ]),

            # layer 5
            # output second scale
            OrderedDict([
                ('head_body_2', vn_layer.HeadBody(input_channels * (2 ** 4 + 2 ** 3))),
            ]),

            # layer 6
            OrderedDict([
                ('trans_2', vn_layer.Transition(input_channels * (2 ** 3))),
            ]),

            # layer 7
            # output first scale, smallest
            OrderedDict([
                ('head_body_3', vn_layer.HeadBody(input_channels * (2 ** 3 + 2 ** 2))),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        stage_6, stage_5, stage_4 = x
        head_body_1 = self.layers[0](stage_6)
        trans_1 = self.layers[1](head_body_1)

        concat_2 = torch.cat([trans_1, stage_5], 1)
        head_body_2 = self.layers[2](concat_2)
        trans_2 = self.layers[3](head_body_2)

        concat_3 = torch.cat([trans_2, stage_4], 1)
        head_body_3 = self.layers[4](concat_3)

        # stage 6, stage 5, stage 4
        features = [head_body_1, head_body_2, head_body_3]
        return features


# PAN+SPP结构
@NECKS.register_module()
class RRYoloV4DarkNeck(nn.Module):
    def __init__(self):
        super(RRYoloV4DarkNeck, self).__init__()
        layer_list = [
            OrderedDict([
                ('head_body0_0', vn_layer.MakeNConv([512, 1024], 1024, 3)),
                ('spp', vn_layer.SpatialPyramidPooling()),
                ('head_body0_1', vn_layer.MakeNConv([512, 1024], 2048, 3)), ]
            ),
            OrderedDict([
                ('trans_0', vn_layer.FuseStage(512)),
                ('head_body1_0', vn_layer.MakeNConv([256, 512], 512, 5))]
            ),

            OrderedDict([
                ('trans_1', vn_layer.FuseStage(256)),
                ('head_body2_1', vn_layer.MakeNConv([128, 256], 256, 5))
            ]),  # 输出0

            OrderedDict([
                ('trans_2', vn_layer.FuseStage(128, is_reversal=True)),
                ('head_body1_1', vn_layer.MakeNConv([256, 512], 512, 5))]
            ),  # 输出1

            OrderedDict([
                ('trans_3', vn_layer.FuseStage(256, is_reversal=True)),
                ('head_body0_2', vn_layer.MakeNConv([512, 1024], 1024, 5))]
            ),  # 输出2
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        out3, out4, out5 = x
        out5 = self.layers[0](out5)
        out4 = self.layers[1]([out4, out5])

        out3 = self.layers[2]([out3, out4])  # 输出0 大图
        out4 = self.layers[3]([out3, out4])  # 输出1
        out5 = self.layers[4]([out4, out5])  # 输出2 小图

        return [out5, out4, out3]

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
