# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from mmdet.models.utils import brick as vn_layer


class TinyYolov4(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.ResConv2dBatchLeaky,)

    def __init__(self, pretrained=False):
        super(TinyYolov4, self).__init__()

        # Network
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

        head = [
            OrderedDict([
                ('10_max', nn.MaxPool2d(2, 2)),
                ('11_conv', vn_layer.Conv2dBatchLeaky(512, 512, 3, 1)),
                ('12_conv', vn_layer.Conv2dBatchLeaky(512, 256, 1, 1)),
            ]),

            OrderedDict([
                ('13_conv', vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('14_conv', nn.Conv2d(512, 3 * (5 + 80), 1)),
            ]),

            OrderedDict([
                ('15_convbatch', vn_layer.Conv2dBatchLeaky(256, 128, 1, 1)),
                ('16_upsample', nn.Upsample(scale_factor=2)),
            ]),

            OrderedDict([
                ('17_convbatch', vn_layer.Conv2dBatchLeaky(384, 256, 3, 1)),
                ('18_conv', nn.Conv2d(256, 3 * (5 + 80), 1)),
            ]),
        ]

        self.backbone = nn.Sequential(backbone)
        self.head = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head])
        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, TinyYolov4.custom_layers)):
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

    def forward(self, x):
        stem, extra_x = self.backbone(x)
        stage0 = self.head[0](stem)
        head0 = self.head[1](stage0)

        stage1 = self.head[2](stage0)
        stage2 = torch.cat((stage1, extra_x), dim=1)
        head1 = self.head[3](stage2)
        head = [head1, head0]
        return head


if __name__ == '__main__':
    import torch

    # darknet 权重路径 https://github.com/AlexeyAB/darknet
    tiny_yolov4_weights_path = '/home/pi/yolo权重/tiny-yolov4/yolov4-tiny.weights'

    tiny_yolov4 = TinyYolov4(pretrained=tiny_yolov4_weights_path)
    new_state_dict = OrderedDict()

    for k, v in tiny_yolov4.state_dict().items():
        if k.startswith('backbone'):
            name = k.replace('backbone', 'backbone.layers')
        else:
            name = k.replace('head', 'bbox_head.layers')
        new_state_dict[name] = v
    data = {"state_dict": new_state_dict}
    # data = {"model": new_state_dict}
    torch.save(data, '../../tiny_yolov4.pth')
