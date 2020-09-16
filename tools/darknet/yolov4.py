# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from mmdet.models.utils import brick as vn_layer


# https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
class Yolov4(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.Resblock_body, vn_layer.Resblock_body.custom_layers, vn_layer.Head,
                     vn_layer.MakeNConv, vn_layer.FuseStage, vn_layer.FuseStage.custom_layers)

    def __init__(self, layers=(1, 2, 8, 8, 4), pretrained=False):
        super(Yolov4, self).__init__()

        num_anchors = (3, 3, 3)
        in_channels_list = (512, 256, 128)
        self.inplanes = 32
        self.aa_conv1 = vn_layer.Conv2dBatchMish(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.aa_stages = nn.ModuleList([
            vn_layer.Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            vn_layer.Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            vn_layer.Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            vn_layer.Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            vn_layer.Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        layer_list1 = [
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
        ]
        head_1 = [
            OrderedDict([
                ('1_head', vn_layer.Head(in_channels_list[0], num_anchors[0], 80)),
            ])
        ]
        layer_list2 = [
            OrderedDict([
                ('trans_2', vn_layer.FuseStage(128, is_reversal=True)),
                ('head_body1_1', vn_layer.MakeNConv([256, 512], 512, 5))]
            )  # 输出1
        ]
        head_2 = [
            OrderedDict([
                ('2_head', vn_layer.Head(in_channels_list[1], num_anchors[1], 80)),
            ])
        ]

        layer_list3 = [
            OrderedDict([
                ('trans_3', vn_layer.FuseStage(256, is_reversal=True)),
                ('head_body0_2', vn_layer.MakeNConv([512, 1024], 1024, 5))]
            )  # 输出2
        ]
        head_3 = [
            OrderedDict([
                ('3_head', vn_layer.Head(in_channels_list[2], num_anchors[2], 80)),
            ]),
        ]

        self.layers1 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list1])
        self.head3 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head_3])
        self.layers2 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list2])
        self.head2 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head_2])
        self.layers3 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list3])
        self.head1 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head_1])
        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, Yolov4.custom_layers)):
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
        return x


if __name__ == '__main__':
    import torch

    # darknet 权重路径 https://github.com/AlexeyAB/darknet
    yolov4_weights_path = '/home/pi/yolo权重/yolov4/yolov4.weights'

    yolov4 = Yolov4(pretrained=yolov4_weights_path)
    new_state_dict = OrderedDict()

    for k, v in yolov4.state_dict().items():
        if k.startswith('aa_'):
            name = k.replace('aa_', 'backbone.')
        elif k.startswith('layers1'):
            name = k.replace('layers1', 'neck.layers')
        elif k.startswith('layers2'):
            name = k.replace('layers2.0', 'neck.layers.3')
        elif k.startswith('layers3'):
            name = k.replace('layers3.0', 'neck.layers.4')
        elif k.startswith('head1'):
            name = k.replace('head1', 'bbox_head.layers')
        elif k.startswith('head2'):
            name = k.replace('head2.0', 'bbox_head.layers.1')
        elif k.startswith('head3'):
            name = k.replace('head3.0', 'bbox_head.layers.2')
        else:
            name = k
        new_state_dict[name] = v
    # print(new_state_dict.keys())
    data = {"state_dict": new_state_dict}
    torch.save(data, '../../yolov4.pth')
