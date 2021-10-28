log_level = 'INFO'
vis_interval = 10  # don't show

logger = dict(type='PyLogging', log_level='info')

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100, interval=50),  # LOW
    dict(type='PeriodicWriterHook', priority=100, interval=vis_interval,
         writers=[dict(type='TensorboardWriter')]),  # LOW
]

evaluator = dict(by_epoch=False,
                 eval_period=500,
                 eval_func=dict(type='COCOEvaluator'))

checkpoint = dict(by_epoch=False, period=500)
