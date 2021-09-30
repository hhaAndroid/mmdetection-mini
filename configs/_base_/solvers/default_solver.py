optimizer = dict(
    type='build_default_optimizer',
    optimizer_cfg=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=None,
)

# TODO: 目前依然不好用，还有很多改进空间
lr_scheduler = dict(type="DefaultLrScheduler",
                    warmup_param_scheduler=dict(type='LinearParamScheduler', start_step=0, end_step=500, start_value=0.001),
                    regular_param_scheduler=dict(type='StepParamScheduler', step=[60000, 80000]),
                    by_epoch=False,
                    warmup_iter_or_epochs=500)

log_level = 'INFO'
runner = dict(type='IterBasedRunner', max_iters=90000)
