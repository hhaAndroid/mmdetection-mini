# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry

__all__ = ['CONV_LAYERS', 'NORM_LAYERS', 'ACTIVATION_LAYERS', 'PADDING_LAYERS', 'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS',
           'DROPOUT_LAYERS', 'POSITIONAL_ENCODING', 'ATTENTION', 'FEEDFORWARD_NETWORK', 'TRANSFORMER_LAYER',
           'TRANSFORMER_LAYER_SEQUENCE']

CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
UPSAMPLE_LAYERS = Registry('upsample layer')
PLUGIN_LAYERS = Registry('plugin layer')

DROPOUT_LAYERS = Registry('drop out layers')
POSITIONAL_ENCODING = Registry('position encoding')
ATTENTION = Registry('attention')
FEEDFORWARD_NETWORK = Registry('feed-forward Network')
TRANSFORMER_LAYER = Registry('transformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')
