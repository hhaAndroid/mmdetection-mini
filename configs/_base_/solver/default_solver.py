optimizer = dict(
    type='build_default_optimizer',
    optimizer_cfg=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=None,
)

lr_scheduler = dict(type="DefaultLrScheduler")
