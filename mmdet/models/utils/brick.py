# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['yolo_init_weight', 'WeightLoader']


def yolo_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class Conv2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),  # , eps=1e-6, momentum=0.01),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


# tiny yolov4专用
class ResConv2dBatchLeaky(nn.Module):

    def __init__(self, in_channels, inter_channels, kernel_size, stride=1, leaky_slope=0.1, return_extra=False):
        super(ResConv2dBatchLeaky, self).__init__()

        self.return_extra = return_extra
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope

        self.layers0 = Conv2dBatchLeaky(self.in_channels // 2, self.inter_channels, self.kernel_size, self.stride,
                                        leaky_slope)
        self.layers1 = Conv2dBatchLeaky(self.inter_channels, self.inter_channels, self.kernel_size, self.stride,
                                        leaky_slope)
        self.layers2 = Conv2dBatchLeaky(self.in_channels, self.in_channels, 1, 1, leaky_slope)

    def forward(self, x):
        y0 = x
        channel = x.shape[1]
        x0 = x[:, channel // 2:, ...]
        x1 = self.layers0(x0)
        x2 = self.layers1(x1)
        x3 = torch.cat((x2, x1), dim=1)
        x4 = self.layers2(x3)
        x = torch.cat((y0, x4), dim=1)
        if self.return_extra:
            return x, x4
        else:
            return x


# yolov4专用
# 有cuda加速实现
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv2dBatchMish(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2dBatchMish, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            Mish()
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


# CSPdarknet内部堆叠的残差块
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            Conv2dBatchMish(channels, hidden_channels, 1),
            Conv2dBatchMish(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


# CSPdarknet的结构块
class Resblock_body(nn.Module):
    custom_layers = (Resblock,)

    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        self.downsample_conv = Conv2dBatchMish(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = Conv2dBatchMish(out_channels, out_channels, 1)
            self.split_conv1 = Conv2dBatchMish(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels // 2),
                Conv2dBatchMish(out_channels, out_channels, 1)
            )
            self.concat_conv = Conv2dBatchMish(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = Conv2dBatchMish(out_channels, out_channels // 2, 1)
            self.split_conv1 = Conv2dBatchMish(out_channels, out_channels // 2, 1)

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels // 2) for _ in range(num_blocks)],
                Conv2dBatchMish(out_channels // 2, out_channels // 2, 1)
            )
            self.concat_conv = Conv2dBatchMish(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


# spp模块
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class MakeNConv(nn.Module):
    def __init__(self, filters_list, in_filters, n):
        super(MakeNConv, self).__init__()
        if n == 3:
            m = nn.Sequential(
                Conv2dBatchLeaky(in_filters, filters_list[0], 1, 1),
                Conv2dBatchLeaky(filters_list[0], filters_list[1], 3, 1),
                Conv2dBatchLeaky(filters_list[1], filters_list[0], 1, 1),
            )
        elif n == 5:
            m = nn.Sequential(
                Conv2dBatchLeaky(in_filters, filters_list[0], 1, 1),
                Conv2dBatchLeaky(filters_list[0], filters_list[1], 3, 1),
                Conv2dBatchLeaky(filters_list[1], filters_list[0], 1, 1),
                Conv2dBatchLeaky(filters_list[0], filters_list[1], 3, 1),
                Conv2dBatchLeaky(filters_list[1], filters_list[0], 1, 1),
            )
        else:
            raise NotImplementedError
        self.m = m

    def forward(self, x):
        return self.m(x)


class Transition(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        half_nchannels = int(nchannels / 2)
        layers = [
            Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            nn.Upsample(scale_factor=2)
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return x


class FuseStage(nn.Module):
    custom_layers = (Transition,)

    def __init__(self, in_filter, is_reversal=False):
        super(FuseStage, self).__init__()
        if is_reversal:
            self.left_conv = Conv2dBatchLeaky(in_filter, in_filter * 2, 3, 2)
            self.right_conv = None
        else:
            self.right_conv = Transition(in_filter)
            self.left_conv = Conv2dBatchLeaky(in_filter, in_filter // 2, 1, 1)

    def forward(self, data):
        left, right = data
        left = self.left_conv(left)
        if self.right_conv:
            right = self.right_conv(right)
        return torch.cat((left, right), dim=1)


# yolov2
class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """

    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super(PaddedMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def __repr__(self):
        return f'{self.__class__.__name__} (kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.kernel_size, self.stride, 0, self.dilation)
        return x


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')
        self.stride = stride
        self.darknet = True

    def __repr__(self):
        return f'{self.__class__.__name__} (stride={self.stride}, darknet_compatible_mode={self.darknet})'

    def forward(self, x):
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        if H % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {H} is not divisible by {self.stride}')
        if W % self.stride != 0:
            raise ValueError(f'Dimension mismatch: {W} is not divisible by {self.stride}')

        # darknet compatible version from: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
        if self.darknet:
            x = x.view(B, C // (self.stride ** 2), H, self.stride, W, self.stride).contiguous()
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
            x = x.view(B, -1, H // self.stride, W // self.stride)
        else:
            ws, hs = self.stride, self.stride
            x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
            x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
            x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
            x = x.view(B, hs * ws * C, H // hs, W // ws)

        return x


class StageBlock(nn.Module):
    custom_layers = ()

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),
            Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1)
        )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):
    custom_layers = (StageBlock, StageBlock.custom_layers)

    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(StageBlock(2 * nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


class HeadBody(nn.Module):
    custom_layers = ()

    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels / 2)
        else:
            half_nchannels = int(nchannels / 3)
        in_nchannels = 2 * half_nchannels
        layers = [
            Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
            Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x


class Head(nn.Module):
    def __init__(self, nchannels, nanchors, nclasses):
        super().__init__()
        mid_nchannels = 2 * nchannels
        layer_list = [
            Conv2dBatchLeaky(nchannels, mid_nchannels, 3, 1),
            nn.Conv2d(mid_nchannels, nanchors * (5 + nclasses), 1, 1, 0),
        ]
        self.feature = nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x


import numpy as np


class WeightLoader:
    """ Load darknet weight files into pytorch layers """

    def __init__(self, filename):
        with open(filename, 'rb') as fp:
            self.header = np.fromfile(fp, count=3, dtype=np.int32).tolist()
            ver_num = self.header[0] * 100 + self.header[1] * 10 + self.header[2]
            print(f'Loading weight file: version {self.header[0]}.{self.header[1]}.{self.header[2]}')

            if ver_num <= 19:
                print(
                    'Weight file uses sizeof to compute variable size, which might lead to undefined behaviour. (choosing int=int32, float=float32)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int32)[0])
            elif ver_num <= 29:
                print(
                    'Weight file uses sizeof to compute variable size, which might lead to undefined behaviour. (choosing int=int32, float=float32, size_t=int64)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])
            else:
                print(
                    'New weight file syntax! Loading of weights might not work properly. Please submit an issue with the weight file version number. [Run with DEBUG logging level]')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])

            self.buf = np.fromfile(fp, dtype=np.float32)

        self.start = 0
        self.size = self.buf.size

    def load_layer(self, layer):
        """ Load weights for a layer from the weights file """
        if type(layer) == nn.Conv2d:
            self._load_conv(layer)
        elif type(layer) == Conv2dBatchLeaky:
            self._load_convbatch(layer)
        elif type(layer) == Conv2dBatchMish:
            self._load_convbatch(layer)
        elif type(layer) == nn.Linear:
            self._load_fc(layer)
        else:
            raise NotImplementedError(f'The layer you are trying to load is not supported [{type(layer)}]')

    def _load_conv(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                              .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_w])
                                .view_as(model.weight.data))
        self.start += num_w

    def _load_convbatch(self, model):
        num_b = model.layers[1].bias.numel()
        model.layers[1].bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                        .view_as(model.layers[1].bias.data))
        self.start += num_b
        model.layers[1].weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                          .view_as(model.layers[1].weight.data))
        self.start += num_b
        model.layers[1].running_mean.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                           .view_as(model.layers[1].running_mean))
        self.start += num_b
        model.layers[1].running_var.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                          .view_as(model.layers[1].running_var))
        self.start += num_b

        num_w = model.layers[0].weight.numel()
        model.layers[0].weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_w])
                                          .view_as(model.layers[0].weight.data))
        self.start += num_w

    def _load_fc(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                              .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_w])
                                .view_as(model.weight.data))
        self.start += num_w


# yolov5所需
# 暂时如此写，后面重构为mmdetection的注册模式来用


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, self.inplace) / 6
        return out


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.01)
        # self.act = nn.Hardswish() if act else nn.Identity() # 需要pytorch1.6支持
        self.act = Hswish() if act else Identity()  # 简单替换掉

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # BN的后面两个参数要非常注意，必须保持一致，否则效果不对劲
        self.bn = nn.BatchNorm2d(2 * c_, eps=0.001, momentum=0.01)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
