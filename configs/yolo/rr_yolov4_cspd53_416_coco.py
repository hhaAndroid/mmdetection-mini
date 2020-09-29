_base_ = './rr_yolov3_d53_416_coco.py'
# model settings
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(type='RRCSPDarknet53'),
    neck=dict(type='RRYoloV4DarkNeck'),
    bbox_head=dict(
        type='RRYolov3Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(142, 110), (192, 243), (459, 401)],
                        [(36, 75), (76, 55), (72, 146)],
                        [(12, 16), (19, 36), (40, 28)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder', scale_x_y=1.05),
        featmap_strides=[32, 16, 8],
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
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum'))
)
