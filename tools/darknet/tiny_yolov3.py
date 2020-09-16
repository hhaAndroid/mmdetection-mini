# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from mmdet.models.utils import brick as vn_layer


class TinyYolov3(nn.Module):

    def __init__(self, pretrained=None):
        super(TinyYolov3, self).__init__()

        # Network
        layer0 = [
            # backbone
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

            # backbone
            OrderedDict([
                ('9_max', nn.MaxPool2d(2, 2)),
                ('10_convbatch', vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('10_zero_pad', nn.ZeroPad2d((0, 1, 0, 1))),
                ('11_max', nn.MaxPool2d(2, 1)),
                ('12_convbatch', vn_layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('13_convbatch', vn_layer.Conv2dBatchLeaky(1024, 256, 1, 1)),
            ]),

        ]

        head0 = [
            OrderedDict([
                ('14_convbatch', vn_layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('15_conv', nn.Conv2d(512, 3 * (5 + 80), 1)),
            ]),

            OrderedDict([
                ('18_convbatch', vn_layer.Conv2dBatchLeaky(256, 128, 1, 1)),
                ('19_upsample', nn.Upsample(scale_factor=2)),
            ]),
            # stage5 / head
            OrderedDict([
                ('21_convbatch', vn_layer.Conv2dBatchLeaky(256 + 128, 256, 3, 1)),
                ('22_conv', nn.Conv2d(256, 3 * (5 + 80), 1)),
            ]),
        ]

        self.layer0 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer0])
        self.head0 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head0])
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

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5, stage4]
        return features


if __name__ == '__main__':
    import torch

    # darknet 权重路径 https://github.com/AlexeyAB/darknet
    tiny_yolov3_weights_path = '/home/pi/yolo权重/tiny-yolov3/yolov3-tiny.weights'

    tiny_yolov3 = TinyYolov3(pretrained=tiny_yolov3_weights_path)
    new_state_dict = OrderedDict()
    for k, v in tiny_yolov3.state_dict().items():
        if k.startswith('layer0'):
            name = k.replace('layer0', 'backbone.layers')
        elif k.startswith('head0'):
            name = k.replace('head0', 'bbox_head.layers')
        else:
            if k.startswith('head1.0'):
                name = k.replace('head1.0', 'bbox_head.layers.1')
            elif k.startswith('head1.1'):
                name = k.replace('head1.1', 'bbox_head.layers.2')
            else:
                name = k.replace('head1', 'bbox_head.layers')
        new_state_dict[name] = v
    data = {"state_dict": new_state_dict}
    torch.save(data, '../../tiny_yolov3.pth')
