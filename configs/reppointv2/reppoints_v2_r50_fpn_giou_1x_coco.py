_base_ = './reppoints_v2_r50_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        loss_bbox_init=dict(_delete_=True, type='GIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(_delete_=True, type='GIoULoss', loss_weight=2.0))
)
