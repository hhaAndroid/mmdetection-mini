# custom_imports = dict(imports=['tools/misc/custom_optimizer.py', 'tools/misc/one_cycle_lr_update.py'])

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/coco_detection.py'
]

detect_mode=False

if detect_mode:
    test_cfg=dict(
        agnostic=False,
        multi_label=False,
        min_bbox_size=0,
        conf_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=300)
else:
    test_cfg=dict(
        agnostic=False,  # 是否区分类别进行 nms，False 表示要区分
        multi_label=True,  # 是否考虑多标签， 单张图检测是为 False，test 时候为 True，可以提高 1 个点的 mAP
        min_bbox_size=0,
        conf_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300)

# model settings
model = dict(
    type='SingleStageDetector',
    # common settings
    comm_cfg=dict(
        pixel_mean=[0, 0, 0],
        pixel_std=[255., 255., 255.],
        to_rgb=True,
        debug=False,
    ),
    backbone=dict(type='YOLOV5Backbone', depth_multiple=0.33, width_multiple=0.5),
    neck=None,
    bbox_head=dict(
        type='YOLOV5Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[0.33, 0.5, 1],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        featmap_strides=[32, 16, 8]
    ),
    test_cfg=test_cfg
)

img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=True)

# dataset settings
dataset_type = 'YOLOV5CocoDataset'
data_root = '/home/hha/dataset/coco/'


train_pipeline = [
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('img_norm_cfg',)),
]


if not detect_mode:
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Yolov5Resize', img_scale=640),
        dict(type='LetterResize', img_scale=(640, 640), scaleup=False, auto=False),
        dict(type='RandomFlip'),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        # 如果不是 debug 模式，可以选择不加载标注
        dict(type='Collect', keys=['img', 'data_sample'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                                     'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                                                     'flip_direction', 'image_id', 'pad_param')),
    ]
else:
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LetterResize', img_scale=(640, 640), scaleup=True, auto=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'], meta_keys=('filename','ori_shape',
                                                      'img_shape', 'pad_shape',
                                                      'scale_factor','pad_param')),
    ]


# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(640, 640),
#         flip=False,
#         transforms=[
#             # dict(type='ShapeLetterResize', img_scale=(640, 640), scaleup=True, auto=False),  # test
#             dict(type='ShapeLetterResize', img_scale=(640, 640), scaleup=False, auto=False, with_yolov5=True),
#             # test，和原始v5完全一致的写法
#             # dict(type='LetterResize', img_scale=(640, 640), scaleup=True, auto=True),  # detect
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'],
#                  meta_keys=('filename', 'ori_filename', 'ori_shape',
#                             'img_shape', 'pad_shape', 'scale_factor', 'flip',
#                             'flip_direction', 'img_norm_cfg', 'pad_param'))
#         ])
# ]

max_epoch = 300

data = dict(
    samples_per_gpu=8,  # total=64 8gpu
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        pad=0.5,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        pad=0.5,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))


evaluation = dict(interval=5, metric='bbox')
checkpoint_config = dict(interval=5)

# optimizer
optimizer = dict(constructor='CustomOptimizer', type='SGD', lr=0.01, momentum=0.937, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# lr_config = dict(policy='OneCycle', repeat_num=repeat_num, max_epoch=max_epoch)
runner = dict(type='EpochBasedRunner', max_epochs=max_epoch)

log_config = dict(interval=30)

custom_hooks = [dict(type='EMAHook', priority=49)]
