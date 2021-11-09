# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

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
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        train_mode=False,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        train_mode=False,
        pipeline=test_pipeline))

dataloader = dict(
    train=dict(type='build_default_dataloader',
               sampler=dict(type="InfiniteSampler"),
               samples_per_gpu=2,
               workers_per_gpu=2),
    val=dict(type='build_default_dataloader',
             sampler=dict(type="InferenceSampler"),
             samples_per_gpu=1,
             workers_per_gpu=0,
             aspect_ratio_grouping=False),
    test=dict(type='build_default_dataloader',
              sampler=dict(type="InferenceSampler"),
              samples_per_gpu=1,
              workers_per_gpu=0,
              aspect_ratio_grouping=False),
)
