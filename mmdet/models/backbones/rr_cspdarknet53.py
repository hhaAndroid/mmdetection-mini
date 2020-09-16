# -*- coding:utf-8 -*-
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from ...cv_core import kaiming_init, constant_init
from ..utils import brick as vn_layer
from ..builder import BACKBONES


@BACKBONES.register_module()
class RRCSPDarknet53(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.Resblock_body, vn_layer.Resblock_body.custom_layers)

    def __init__(self, layers=(1, 2, 8, 8, 4), input_channels=32, pretrained=None):
        super().__init__()
        self.inplanes = input_channels
        self.conv1 = vn_layer.Conv2dBatchMish(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            vn_layer.Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            vn_layer.Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            vn_layer.Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            vn_layer.Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            vn_layer.Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, RRCSPDarknet53.custom_layers)):
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
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return [out3, out4, out5]  # 由大到小特征图输出

