_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/dataloaders/default_dataloader.py',
    '../_base_/solvers/default_solver.py'
]

logger = dict(type='PyLogging', log_level='info')

runner = dict(type='IterBasedRunner', max_iters=1000)
