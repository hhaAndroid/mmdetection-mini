_base_ = './rr_yolov3_d53_416_coco.py'
# model settings

_yolo_type = 0  # 0=5s 1=5m 2=5l 3=5x
if _yolo_type == 0:
    _depth_multiple = 0.33
    _width_multiple = 0.5
elif _yolo_type == 1:
    _depth_multiple = 0.67
    _width_multiple = 0.75
elif _yolo_type == 2:
    _depth_multiple = 1.0
    _width_multiple = 1.0
elif _yolo_type == 3:
    _depth_multiple = 1.33
    _width_multiple = 1.25
else:
    raise NotImplementedError

model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(type='RRYoloV5Backbone', depth_multiple=_depth_multiple, width_multiple=_width_multiple),
    neck=None,
    bbox_head=dict(
        type='RRYolov5Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[_depth_multiple, _width_multiple],  # yolov5s
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOV5BBoxCoder'),
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

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='LetterResize', img_scale=(640, 640)),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'pad_param'))
        ])
]

data = dict(
    test=dict(pipeline=test_pipeline))


test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.0000001,
    conf_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=300)  # 1000
