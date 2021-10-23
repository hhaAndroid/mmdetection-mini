optimizer = dict(
    type='build_default_optimizer',
    optimizer_cfg=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=None,
)


lr_scheduler=dict(type='build_default_lr_scheduler',
                  param_scheduler=[
                      dict(type='LinearParamScheduler', begin=0, end=500, start_value=0.001, by_epoch=False),
                      dict(type='StepParamScheduler', begin=500, end=100000,step=[60000, 80000], by_epoch=False)])


runner = dict(type='IterBasedRunner', max_iters=90000)
