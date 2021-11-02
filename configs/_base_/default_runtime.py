log_level = 'INFO'
vis_interval = dict(train=10, val=20)

logger = dict(type='PyLogging', log_level='info')
writer = [dict(type='WandbWriter', init_kwargs=dict(project='demo', entity="huanghaian"))]

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100, interval=50),  # LOw
]

evaluator = dict(type='COCOEvaluator')
checkpoint = dict(by_epoch=False, period=1000)
workflow = [('train', 100), ('val', 1)]
