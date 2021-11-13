_base_ = [
    './yolov5s_v6.py'
]

# model settings
model = dict(
    backbone=dict(depth_multiple=1.33, width_multiple=1.25),
   bbox_head=dict(out_channels=[1.33, 1.25, 1])
)
