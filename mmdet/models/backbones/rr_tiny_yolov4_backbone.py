# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from ...cv_core import kaiming_init, constant_init
from ..utils import brick as vn_layer
from ..builder import BACKBONES


@BACKBONES.register_module()
class RRTinyYolov4Backbone(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.ResConv2dBatchLeaky,)

    def __init__(self, pretrained=None):
        super(RRTinyYolov4Backbone, self).__init__()

        backbone = OrderedDict([
            ('0_convbatch', vn_layer.Conv2dBatchLeaky(3, 32, 3, 2)),
            ('1_convbatch', vn_layer.Conv2dBatchLeaky(32, 64, 3, 2)),
            ('2_convbatch', vn_layer.Conv2dBatchLeaky(64, 64, 3, 1)),
            ('3_resconvbatch', vn_layer.ResConv2dBatchLeaky(64, 32, 3, 1)),
            ('4_max', nn.MaxPool2d(2, 2)),
            ('5_convbatch', vn_layer.Conv2dBatchLeaky(128, 128, 3, 1)),
            ('6_resconvbatch', vn_layer.ResConv2dBatchLeaky(128, 64, 3, 1)),
            ('7_max', nn.MaxPool2d(2, 2)),
            ('8_convbatch', vn_layer.Conv2dBatchLeaky(256, 256, 3, 1)),
            ('9_resconvbatch', vn_layer.ResConv2dBatchLeaky(256, 128, 3, 1, return_extra=True)),
        ])

        self.layers = nn.Sequential(backbone)
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
        if pretrained is not None:
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
        stem, extra_x = self.layers(x)
        return [stem, extra_x]
