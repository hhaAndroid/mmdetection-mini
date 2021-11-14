_base_ = './retinanet_r50_fpn_1x_iter_coco.py'


dataloader = dict(
    train=dict(
        sampler=dict(type="EpochBaseSampler"))
)


lr_scheduler = dict(type='build_default_lr_scheduler',
                    param_steps=[0, 1000],
                    by_epoch=False,
                    param_scheduler=[
                        dict(type='LinearParamScheduler', start_value=0.001),
                        dict(type='StepParamScheduler',
                             step=[8, 11], by_epoch=True)])

runner = dict(_delete_=True,type='EpochBasedRunner', max_epochs=12)

checkpoint = dict(by_epoch=True, period=1)
workflow = [('train', 1), ('val', 1)]


