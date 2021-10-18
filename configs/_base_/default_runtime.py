log_level = 'INFO'

logger = dict(type='PyLogging', log_level='info')

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100),  # LOW
]

evaluator = dict(by_epoch=False,
                 eval_period=10,
                 eval_func=dict(type='COCOEvaluator'))


