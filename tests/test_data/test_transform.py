from mmdet import cv_core
from mmdet.datasets import PIPELINES
import numpy as np


def resize_test():
    #
    transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
    transform = cv_core.build_from_cfg(transform, PIPELINES)
    input_shape = (60, 84, 3)
    img = np.zeros(input_shape, dtype=np.uint8)
    output = transform(dict(img=img))
    print(output['img_shape'])

    transform = dict(
        type='Resize',
        img_scale=(1333, 800),
        ratio_range=(0.9, 1.1),
        keep_ratio=True)
    transform = cv_core.build_from_cfg(transform, PIPELINES)
    input_shape = (60, 84, 3)
    img = np.zeros(input_shape, dtype=np.uint8)
    output = transform(dict(img=img))
    print(output['img_shape'])


if __name__ == '__main__':
    resize_test()
