_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/solvers/default_solver.py',
    '../_base_/default_runtime.py'
]

# runtime_func = dict(type='VisFuncStorage', train_vis_interval=-1, val_vis_interval=-1)
