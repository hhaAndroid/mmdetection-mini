log_level = 'INFO'
vis_interval=dict(train=500,val=20)

logger = dict(type='PyLogging', log_level='info')

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100, interval=50),  # LOW
    dict(type='PeriodicWriterHook', priority=100, interval=vis_interval,
         writers=[dict(type='TensorboardWriter')]),  # LOW
]

# test_custom_hooks = [
#     dict(type='DefaultLoggerHook', priority=100, interval=1),  # LOW
#     dict(type='PeriodicWriterHook', priority=100, interval=vis_interval,
#          writers=[dict(type='TensorboardWriter')]),  # LOW
# ]

test_custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100, interval=1),  # LOW
    dict(type='PeriodicWriterHook', priority=100, interval=vis_interval,
         writers=[dict(type='WandbWriter',init_kwargs=dict(project='onestage',entity="huanghaian"))]),  # LOW
]


evaluator = dict(type='COCOEvaluator')
checkpoint = dict(by_epoch=False, period=1000)
workflow=[('train',1000), ('val',1)]
