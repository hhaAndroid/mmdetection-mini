optimizer = dict(
    type='build_default_optimizer',
    optimizer_cfg=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=None,
)

lr_scheduler = dict(type="DefaultLrScheduler",
                    warmup_param_scheduler=dict(type='ConstantParamScheduler', value=0.1),
                    regular_param_scheduler=dict(type='ConstantParamScheduler', value=0.2),
                    by_epoch=False,
                    warmup_iter_or_epochs=100)

log_level = 'INFO'
