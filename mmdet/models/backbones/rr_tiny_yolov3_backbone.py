# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from ...cv_core import kaiming_init, constant_init
from ..utils import brick as vn_layer
from ..builder import BACKBONES


@BACKBONES.register_module()
class RRTinyYolov3Backbone(nn.Module):

    def __init__(self, pretrained=None):
        super(RRTinyYolov3Backbone, self).__init__()

        # Network
        layer_list = [
            OrderedDict([
                ('0_convbatch', vn_layer.Conv2dBatchLeaky(3, 16, 3, 1)),
                ('1_max', nn.MaxPool2d(2, 2)),
                ('2_convbatch', vn_layer.Conv2dBatchLeaky(16, 32, 3, 1)),
                ('3_max', nn.MaxPool2d(2, 2)),
                ('4_convbatch', vn_layer.Conv2dBatchLeaky(32, 64, 3, 1)),
            ]),

            OrderedDict([
                ('5_max', nn.MaxPool2d(2, 2)),
                ('6_convbatch', vn_layer.Conv2dBatchLeaky(64, 128, 3, 1)),
            ]),

            OrderedDict([
                ('7_max', nn.MaxPool2d(2, 2)),
                ('8_convbatch', vn_layer.Conv2dBatchLeaky(128, 256, 3, 1)),
            ]),

            OrderedDict([
                ('9_max', nn.MaxPool2d(2, 2)),
                ('10_convbatch', vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('10_zero_pad', nn.ZeroPad2d((0, 1, 0, 1))),
                ('11_max', nn.MaxPool2d(2, 1)),
                ('12_convbatch', vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('13_convbatch', vn_layer.Conv2dBatchLeaky(1024, 256, 1, 1)),
            ]),

        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.__modules_recurse(module)
            else:
                yield module

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            weights = vn_layer.WeightLoader(pretrained)
            for module in self.__modules_recurse():
                try:
                    weights.load_layer(module)
                    print(f'Layer loaded: {module}')
                    if weights.start >= weights.size:
                        print(f'Finished loading weights [{weights.start}/{weights.size} weights]')
                        break
                except NotImplementedError:
                    print(f'Layer skipped: {module.__class__.__name__}')
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5]
        return features
