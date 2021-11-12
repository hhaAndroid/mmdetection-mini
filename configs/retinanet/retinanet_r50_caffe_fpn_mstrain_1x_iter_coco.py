_base_ = './retinanet_r50_fpn_1x_iter_coco.py'

model = dict(
    comm_cfg=dict(
        pixel_mean=[103.530, 116.280, 123.675],
        pixel_std=[1.0, 1.0, 1.0],
        to_rgb=False,
        debug=False,
    ),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))

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

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_sample']),
]

val_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # 如果不是 debug 模式，可以选择不加载标注
    dict(type='Collect', keys=['img', 'data_sample']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # 如果不是 debug 模式，可以选择不加载标注
    dict(type='Collect', keys=['img', 'data_sample']),
]


data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))
