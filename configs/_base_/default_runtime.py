log_level = 'INFO'

logger = dict(type='PyLogging', log_level='info')
writer = [dict(type='LocalWriter', show=True)]
# writer = [dict(type='WandbWriter', init_kwargs=dict(project='demo', entity="huanghaian"))]

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100, interval=50),  # L0w
]

evaluator = dict(type='COCOEvaluator')
checkpoint = dict(by_epoch=False, period=200)
workflow = [('train', 200), ('val', 1)]
