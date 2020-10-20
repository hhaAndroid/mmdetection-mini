# model settings
model = dict(
    type='SingleStageDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  # 固定stem和第0个stage
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,  # 除了frozen_stages外的其他bn都采用全局均值和方差，但是可训练参数依然可以更新
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,  # 每层特征图的base anchor scale,如果变大，则整体anchor都会放大
            scales_per_octave=3,   # 每层有3个尺度 2**0 2**(1/3) 2**(2/3)
            ratios=[0.5, 1.0, 2.0],  # 每层的anchor有3种长宽比 故每一层每个位置有9个anchor
            strides=[8, 16, 32, 64, 128]),  # 每个特征图层输出stride,故anchor范围是4x8=32,4x128x2**(2/3)=812.7
        bbox_coder=dict(  # 基于anchor的中心点平移，wh缩放预测，编解码函数
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))  # 注意不是smooth l1，而是l1(mmdet改了，说效果更好)，原论文是sl1
# training and testing settings
train_cfg = dict(
    # 双阈值策略，下列设置会出现忽略样本，且可能引入低质量anchor
    # min_pos_iou=0 表示每个gt都一定有anchor匹配
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,  # 遍历anchor,anchor和所有gt的最大iou大于pos_iou_thr，则该anchor是正样本且最大iou对应的gt是匹配对象
        neg_iou_thr=0.4,  # 遍历anchor,anchor和所有gt的最大iou小于neg_iou_thr，则该anchor是背景
        min_pos_iou=0,  # 遍历gt，gt和所有anchor的最大iou大于min_pos_iou，则该对应的anchor也是正样本，负责对于gt的匹配
        ignore_iof_thr=-1),
    allowed_border=-1,  # 边界位置anchor不排除，通过其他方式排除
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
