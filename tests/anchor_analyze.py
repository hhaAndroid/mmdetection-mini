from mmdet.det_core import build_anchor_generator
from mmdet import cv_core
import numpy as np

if __name__ == '__main__':
    input_shape_hw = (500, 500, 3)
    img = np.zeros(input_shape_hw)
    stride = 10
    feature_map = [input_shape_hw[0] // stride, input_shape_hw[1] // stride]
    # anchor_generator_cfg = dict(
    #     type='AnchorGenerator',
    #     scales=[1],  # 缩放系数
    #     ratios=[0.5, 1.0, 2.0],  # 宽高比例
    #     strides=[stride])  # 特征图相对原图下降比例
    # retinanet设置
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[stride])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    anchor = anchor_generator.grid_anchors([feature_map])[0].cpu().numpy()
    print(anchor)  # 输出原图尺度上anchor坐标 xyxy格式 左上角格式
    anchor = np.random.permutation(anchor)
    cv_core.show_bbox(img, anchor[:100], thickness=1)

