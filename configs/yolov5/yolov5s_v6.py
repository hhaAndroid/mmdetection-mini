_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/coco_detection.py'
]

use_ceph = False

if use_ceph:
    file_client_args = dict(
        backend='petrel',
        path_mapping=dict({
            './data/': 's3://openmmlab/datasets/detection/',
            'data/': 's3://openmmlab/datasets/detection/'
        }))
else:
    file_client_args = dict(backend='disk')



detect_mode = False

if detect_mode:
    test_cfg = dict(
        agnostic=False,
        multi_label=False,
        min_bbox_size=0,
        conf_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=300)
else:
    test_cfg = dict(
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

# dataset settings
dataset_type = 'YOLOV5CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_sample'],
         meta_keys=('filename', 'ori_filename', 'img_shape', 'image_id')),
]

if not detect_mode:
    test_pipeline = [
        dict(type='LoadImageFromFile',file_client_args=file_client_args),
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
        dict(type='LoadImageFromFile',file_client_args=file_client_args),
        dict(type='LetterResize', img_scale=(640, 640), scaleup=True, auto=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape',
                                                      'img_shape', 'pad_shape',
                                                      'scale_factor', 'pad_param')),
    ]


data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline,
        use_ceph=use_ceph,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        pad=0.5,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        use_ceph=use_ceph),
    test=dict(
        type=dataset_type,
        pad=0.5,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        use_ceph=use_ceph))

dataloader = dict(train=dict(sampler=dict(type="EpochBaseSampler"),
                             aspect_ratio_grouping=False,
                             samples_per_gpu=16,
                             workers_per_gpu=4))

# optimizer
optimizer = dict(
    type='build_yolov5_optimizer',
    optimizer_cfg=dict(type='SGD', lr=0.01, momentum=0.937, nesterov=True),
    weight_decay=0.0005
)

lr_scheduler = dict(type='build_default_lr_scheduler',
                    param_steps=[0, 1000],
                    by_epoch=False,
                    param_scheduler=[
                        dict(type='Yolov5WramUpParamScheduler'),
                        dict(type='Yolov5OneCycleParamScheduler', begin=0, by_epoch=True)])

runner = dict(type='EpochBasedRunner', max_epochs=300)

checkpoint = dict(by_epoch=True, period=10, max_to_keep=2)
workflow = [('train', 10), ('val', 1)]

custom_hooks = [
    dict(type='ExpMomentumEMAHook', priority=49),
    dict(type='Yolov5LoggerHook', priority=100, interval=50),  # Low
]
