# -*- coding:utf-8 -*-
import math
import functools
import torch

import torch.nn as nn
from .base_yolo_head import BaseYOLOHead
from ..utils import brick as vn_layer
from ..builder import HEADS


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(_make_divisible, divisor=divisor, width_multiple=width_multiple)


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


# 可能写的不太优美
@HEADS.register_module()
class RRYolov5Head(BaseYOLOHead):
    # 为了不传入新的参数，默认将self.out_channels=[depth_multiple,width_multiple]

    def _init_layers(self):
        model = []

        make_div8_fun = make_divisible(8, self.out_channels[1])
        make_round_fun = make_round(self.out_channels[0])

        conv1 = vn_layer.Conv(make_div8_fun(1024), make_div8_fun(512))
        model.append(conv1)  # 0
        up1 = nn.Upsample(scale_factor=2)
        model.append(up1)  # 1
        cont1 = vn_layer.Concat()
        model.append(cont1)  # 2
        bsp1 = vn_layer.BottleneckCSP(make_div8_fun(512)+make_div8_fun(self.in_channels[0]), make_div8_fun(512), make_round_fun(3), shortcut=False)
        model.append(bsp1)  # 3

        conv2 = vn_layer.Conv(make_div8_fun(512), make_div8_fun(256))
        model.append(conv2)  # 4
        up2 = nn.Upsample(scale_factor=2)
        model.append(up2)  # 5
        cont2 = vn_layer.Concat()
        model.append(cont2)  # 6
        bsp2 = vn_layer.BottleneckCSP(make_div8_fun(256)+make_div8_fun(self.in_channels[1]), make_div8_fun(256), make_round_fun(3), shortcut=False)
        model.append(bsp2)  # 7

        conv3 = vn_layer.Conv(make_div8_fun(256), make_div8_fun(256), k=3, s=2)
        model.append(conv3)  # 8
        cont3 = vn_layer.Concat()
        model.append(cont3)  # 9
        bsp3 = vn_layer.BottleneckCSP(make_div8_fun(256)+make_div8_fun(256), make_div8_fun(512), make_round_fun(3), shortcut=False)
        model.append(bsp3)  # 10

        conv4 = vn_layer.Conv(make_div8_fun(512), make_div8_fun(512), k=3, s=2)
        model.append(conv4)  # 11
        cont4 = vn_layer.Concat()
        model.append(cont4)  # 12
        bsp4 = vn_layer.BottleneckCSP(make_div8_fun(1024), make_div8_fun(1024), make_round_fun(3), shortcut=False)
        model.append(bsp4)  # 13

        self.det = nn.Sequential(*model)
        self.head = nn.Sequential(
            nn.Conv2d(make_div8_fun(256), 255, 1),
            nn.Conv2d(make_div8_fun(512), 255, 1),
            nn.Conv2d(make_div8_fun(1024), 255, 1),
        )

    def forward(self, feats):
        large_feat, inter_feat, small_feat = feats

        small_feat = self.det[0](small_feat)
        x = self.det[1](small_feat)
        x = self.det[2]([x, inter_feat])
        x = self.det[3](x)
        inter_feat = self.det[4](x)

        x = self.det[5](inter_feat)
        x = self.det[6]([x, large_feat])
        x = self.det[7](x)  # 128
        out0 = self.head[0](x)  # 第一个输出层

        x = self.det[8](x)
        x = self.det[9]([x, inter_feat])
        x = self.det[10](x)  #
        out1 = self.head[1](x)  # 第二个输出层

        x = self.det[11](x)
        x = self.det[12]([x, small_feat])
        x = self.det[13](x)  # 256
        out2 = self.head[2](x)  # 第三个输出层

        return tuple([out2, out1, out0]),  # 从小到大特征图返回
