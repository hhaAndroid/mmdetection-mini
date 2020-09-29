_base_ = './rr_yolov3_d53_416_coco.py'
# model settings
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(type='RRTinyYolov4Backbone'),
    neck=None,
    bbox_head=dict(
        type='RRTinyYolov4Head',
        num_classes=80,
        in_channels=[512, 256],
        out_channels=[256, 128],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(81, 82), (135, 169), (344, 319)],
                        [(23, 27), (37, 58), (81, 82)]],
            strides=[32, 16]),
        bbox_coder=dict(type='YOLOBBoxCoder', scale_x_y=1.05),
        featmap_strides=[32, 16],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')))
