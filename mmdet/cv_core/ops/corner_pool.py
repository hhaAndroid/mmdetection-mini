import torch
from torch import nn
from torch.autograd import Function
from ..cnn import ConvModule
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'top_pool_forward', 'top_pool_backward', 'bottom_pool_forward',
    'bottom_pool_backward', 'left_pool_forward', 'left_pool_backward',
    'right_pool_forward', 'right_pool_backward'
])


class TopPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = ext_module.top_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.top_pool_backward(input, grad_output)
        return output


class BottomPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = ext_module.bottom_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.bottom_pool_backward(input, grad_output)
        return output


class LeftPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = ext_module.left_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.left_pool_backward(input, grad_output)
        return output


class RightPoolFunction(Function):

    @staticmethod
    def forward(ctx, input):
        output = ext_module.right_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.right_pool_backward(input, grad_output)
        return output


class CornerPool(nn.Module):
    """Corner Pooling.

    Corner Pooling is a new type of pooling layer that helps a
    convolutional network better localize corners of bounding boxes.

    Please refer to https://arxiv.org/abs/1808.01244 for more details.
    Code is modified from https://github.com/princeton-vl/CornerNet-Lite.

    Args:
        mode(str): Pooling orientation for the pooling layer

            - 'bottom': Bottom Pooling
            - 'left': Left Pooling
            - 'right': Right Pooling
            - 'top': Top Pooling

    Returns:
        Feature map after pooling.
    """

    pool_functions = {
        'bottom': BottomPoolFunction,
        'left': LeftPoolFunction,
        'right': RightPoolFunction,
        'top': TopPoolFunction,
    }

    cummax_dim_flip = {
        'bottom': (2, False),
        'left': (3, True),
        'right': (3, False),
        'top': (2, True),
    }

    def __init__(self, mode):
        super(CornerPool, self).__init__()
        assert mode in self.pool_functions
        self.mode = mode
        self.corner_pool = self.pool_functions[mode]

    def forward(self, x):
        if torch.__version__ != 'parrots' and torch.__version__ >= '1.5.0':
            dim, flip = self.cummax_dim_flip[self.mode]
            if flip:
                x = x.flip(dim)
            pool_tensor, _ = torch.cummax(x, dim=dim)
            if flip:
                pool_tensor = pool_tensor.flip(dim)
            return pool_tensor
        else:
            return self.corner_pool.apply(x)


class CornerPoolPack(nn.Module):
    def __init__(self, dim, pool1, pool2, conv_cfg=None, norm_cfg=None, first_kernel_size=3, kernel_size=3,
                 corner_dim=128):
        super(CornerPoolPack, self).__init__()
        self.p1_conv1 = ConvModule(
            dim,
            corner_dim,
            first_kernel_size,
            stride=1,
            padding=(first_kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.p2_conv1 = ConvModule(
            dim,
            corner_dim,
            first_kernel_size,
            stride=1,
            padding=(first_kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.p_conv1 = nn.Conv2d(corner_dim, dim, 3, padding=1, bias=False)
        self.p_gn1 = nn.GroupNorm(num_groups=32, num_channels=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvModule(
            dim,
            dim,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.pool1 = pool1
        self.pool2 = pool2

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_gn1 = self.p_gn1(p_conv1)

        conv1 = self.conv1(x)
        gn1 = self.gn1(conv1)
        relu1 = self.relu1(p_gn1 + gn1)

        conv2 = self.conv2(relu1)
        return conv2


class TLPool(CornerPoolPack):
    def __init__(self, dim, conv_cfg=None, norm_cfg=None, first_kernel_size=3, kernel_size=3, corner_dim=128):
        super(TLPool, self).__init__(dim, CornerPool('top'), CornerPool('left'), conv_cfg, norm_cfg, first_kernel_size,
                                     kernel_size, corner_dim)


class BRPool(CornerPoolPack):
    def __init__(self, dim, conv_cfg=None, norm_cfg=None, first_kernel_size=3, kernel_size=3, corner_dim=128):
        super(BRPool, self).__init__(dim, CornerPool('bottom'), CornerPool('right'), conv_cfg, norm_cfg,
                                     first_kernel_size, kernel_size, corner_dim)
