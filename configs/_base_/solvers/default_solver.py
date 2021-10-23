optimizer = dict(
    type='build_default_optimizer',
    optimizer_cfg=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=None,
)


lr_scheduler=dict(type='build_default_lr_scheduler',
                  param_steps=[0,500],
                  by_epoch=False,
                  param_scheduler=[
                      dict(type='LinearParamScheduler', start_value=0.001),
                      dict(type='StepParamScheduler', step=[60000, 80000])])


runner = dict(type='IterBasedRunner', max_iters=90000)
