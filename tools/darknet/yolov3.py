# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from mmdet.models.utils import brick as vn_layer


class YOLOV3(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.Stage, vn_layer.Stage.custom_layers, vn_layer.HeadBody, vn_layer.Transition,
                     vn_layer.Head)

    def __init__(self, pretrained=False, input_channels=32):
        super(YOLOV3, self).__init__()
        stage_cfg = {'stage_2': 2, 'stage_3': 3, 'stage_4': 9, 'stage_5': 9, 'stage_6': 5}

        # darknet53
        darknet = [
            # layer 0
            # first scale, smallest
            OrderedDict([
                ('stage_1', vn_layer.Conv2dBatchLeaky(3, input_channels, 3, 1)),  # 32
                ('stage_2', vn_layer.Stage(input_channels, stage_cfg['stage_2'])),  # 32-->64
                ('stage_3', vn_layer.Stage(input_channels * (2 ** 1), stage_cfg['stage_3'])),  # 64-->128
                ('stage_4', vn_layer.Stage(input_channels * (2 ** 2), stage_cfg['stage_4'])),  # 128-->256
            ]),

            # layer 1
            # second scale
            OrderedDict([
                ('stage_5', vn_layer.Stage(input_channels * (2 ** 3), stage_cfg['stage_5'])),  # 256-->512
            ]),

            # layer 2
            # third scale, largest
            OrderedDict([
                ('stage_6', vn_layer.Stage(input_channels * (2 ** 4), stage_cfg['stage_6'])),  # 512-->1024
            ]),
        ]

        self.backbone = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in darknet])

        in_channels_list = (512, 256, 128)
        num_anchors = (3, 3, 3)
        num_classes = 80

        # head
        layer_list1 = [
            OrderedDict([
                ('head_body_1', vn_layer.HeadBody(input_channels * (2 ** 5), first_head=True)),
            ]),
        ]

        head_1 = [
            OrderedDict([
                ('1_head', vn_layer.Head(in_channels_list[0], num_anchors[0], num_classes)),
            ])
        ]

        layer_list2 = [
            # layer 4
            OrderedDict([
                ('trans_1', vn_layer.Transition(input_channels * (2 ** 4))),
            ]),

            # layer 5
            # output second scale
            OrderedDict([
                ('head_body_2', vn_layer.HeadBody(input_channels * (2 ** 4 + 2 ** 3))),
            ])
        ]

        head_2 = [
            OrderedDict([
                ('2_head', vn_layer.Head(in_channels_list[1], num_anchors[1], num_classes)),
            ])
        ]

        layer_list3 = [
            # layer 6
            OrderedDict([
                ('trans_2', vn_layer.Transition(input_channels * (2 ** 3))),
            ]),

            # layer 7
            # output first scale, smallest
            OrderedDict([
                ('head_body_3', vn_layer.HeadBody(input_channels * (2 ** 3 + 2 ** 2))),
            ])
        ]

        head_3 = [

            OrderedDict([
                ('3_head', vn_layer.Head(in_channels_list[2], num_anchors[2], num_classes)),
            ]),

        ]

        self.layers1 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list1])
        self.head1 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head_1])

        self.layers2 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list2])
        self.head2 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head_2])

        self.layers3 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list3])
        self.head3 = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head_3])

        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, YOLOV3.custom_layers)):
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
        stage_4 = self.layers[0](x)
        stage_5 = self.layers[1](stage_4)
        stage_6 = self.layers[2](stage_5)
        return [stage_6, stage_5, stage_4]  # 由小到大特征图输出


if __name__ == '__main__':
    import torch

    # darknet 权重路径 https://github.com/AlexeyAB/darknet
    yolov3_weights_path = '/home/pi/yolo权重/yolov3/yolov3.weights'
    
    yolov3 = YOLOV3(pretrained=yolov3_weights_path)
    new_state_dict = OrderedDict()
    for k, v in yolov3.state_dict().items():
        if k.startswith('backbone'):
            name = k.replace('backbone', 'backbone.layers')
        elif k.startswith('layers1'):
            name = k.replace('layers1', 'neck.layers')
        elif k.startswith('layers2.0'):
            name = k.replace('layers2.0', 'neck.layers.1')
        elif k.startswith('layers2.1'):
            name = k.replace('layers2.1', 'neck.layers.2')
        elif k.startswith('layers3.0'):
            name = k.replace('layers3.0', 'neck.layers.3')
        elif k.startswith('layers3.1'):
            name = k.replace('layers3.1', 'neck.layers.4')
        elif k.startswith('head1'):
            name = k.replace('head1', 'bbox_head.layers')
        elif k.startswith('head2'):
            name = k.replace('head2.0', 'bbox_head.layers.1')
        elif k.startswith('head3'):
            name = k.replace('head3.0', 'bbox_head.layers.2')
        else:
            name = k
        new_state_dict[name] = v
    data = {"state_dict": new_state_dict}
    torch.save(data, '../../yolov3.pth')
