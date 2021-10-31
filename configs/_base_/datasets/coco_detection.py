import os

# dataset settings
dataset_type = 'CocoDataset'
X_DATA = os.getenv('X_DATA', 'data/')
# data_root = X_DATA + 'coco/'
data_root = '/home/hha/dataset/project/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(416, 416), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_sample']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(416, 416), keep_ratio=True),
    dict(type='RandomFlip'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_sample']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(416, 416), keep_ratio=True),
    dict(type='RandomFlip'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_sample']),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(416, 416),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

classes = ('out-ok', 'out-ng')
data = dict(
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        train_mode=False,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        train_mode=False,
        pipeline=test_pipeline))
