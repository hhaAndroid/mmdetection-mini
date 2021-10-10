_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/dataloaders/default_dataloader.py',
    '../_base_/solvers/default_solver.py'
]

logger = dict(type='PyLogging', log_level='info')

custom_hooks = [
    dict(type='DefaultLoggerHook', priority=100),  # LOW
]

evaluator = dict(by_epoch=False,
                 eval_period=100,
                 eval_func=dict(type='COCOEvaluator'))
