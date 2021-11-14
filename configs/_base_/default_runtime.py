traing_mode = 'cuda'  # cuda cpu
log_level = 'INFO'

logger = dict(type='PyLogging', log_level='debug')
writer = [dict(type='LocalWriter', show=True)]
# writer = [dict(type='WandbWriter', init_kwargs=dict(project='demo', entity="huanghaian"))]

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100, interval=1),  # L0w
]

evaluator = dict(type='COCOEvaluator')
checkpoint = dict(by_epoch=False, period=5000)
workflow = [('train', 5000), ('val', 1)]
