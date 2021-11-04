# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

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


dataloader = dict(
    train=dict(type='build_default_dataloader',
               sampler=dict(type="InfiniteSampler"),
               samples_per_gpu=4,
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
