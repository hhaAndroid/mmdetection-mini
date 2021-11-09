from .default_func_storage import DefaultFuncStorage
from .builder import FUNCSTORAGES
from ..visualizer import get_event_storage
import torch
import copy
from cvcore import convert_image_to_rgb, master_only

__all__ = ['VisFuncStorage']


@FUNCSTORAGES.register_module()
class VisFuncStorage(DefaultFuncStorage):
    def __init__(self, train_vis_interval=-1, val_vis_interval=-1):
        self.train_vis_interval = train_vis_interval
        self.val_vis_interval = val_vis_interval

    @master_only
    def visualize_training(self, kwargs):
        if self.train_vis_interval > 0:
            model = kwargs['model']
            cls_scores = kwargs['cls_scores']
            bbox_preds = kwargs['bbox_preds']
            batched_inputs = kwargs['batched_inputs']

            storage = get_event_storage()

            if storage.iter % self.train_vis_interval == 0:

                with torch.no_grad():
                    imgs = []
                    data_samples = []
                    results = model.get_bboxes(cls_scores, bbox_preds, batched_inputs, skip_post=True)
                    for (input, result) in zip(batched_inputs, results):
                        imgs.append(input['img'])
                        data_sample = input["data_sample"]
                        data_sample.pred_instances = result.to('cpu')
                        data_samples.append(data_sample)

                    for (img, data_sample) in zip(imgs, data_samples):
                        vis_img = convert_image_to_rgb(img.permute(1, 2, 0), "RGB")
                        vis_name = "TRAIN:GT--Predicted"
                        storage.add_image(vis_name, vis_img, data_sample)
                        break  # only visualize one image in a batch

    @master_only
    def visualize_val(self, kwargs):
        if self.val_vis_interval > 0:

            results = kwargs['results']
            batched_inputs = kwargs['batched_inputs']

            storage = get_event_storage()
            if storage.iter % self.val_vis_interval == 0:
                with torch.no_grad():
                    imgs = []
                    data_samples = []
                    for (input, result) in zip(batched_inputs, results):
                        # result=result.new() # TODO
                        result = copy.deepcopy(result)  # TODO
                        imgs.append(input['img'])
                        scale_factor = input['img_metas']['scale_factor']
                        result.bboxes.scale(scale_factor[0], scale_factor[1])

                        data_sample = input["data_sample"]
                        data_sample.pred_instances = result.to('cpu')
                        data_samples.append(data_sample)

                    for (img, data_sample) in zip(imgs, data_samples):
                        vis_img = convert_image_to_rgb(img.permute(1, 2, 0), "RGB")
                        vis_name = "VAL:GT--Predicted"
                        storage.add_image(vis_name, vis_img, data_sample)
                        break  # only visualize one image in a batch

