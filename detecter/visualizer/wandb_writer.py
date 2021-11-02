from .base_writer import BaseWriter
from .builder import WRITERS
from detecter.core.structures import BoxMode
import torch

__all__ = ['WandbWriter']


@WRITERS.register_module()
class WandbWriter(BaseWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, init_kwargs=None, commit=True, with_step=False, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step
        self.kwargs = kwargs
        # 必须全局自增长，和 tensorboard 不一样，with_step 设置为 False 也一样
        # 目前解决办法就是 log 中不存储 step 信息
        self._step = 1
        self.image_table = None

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def init(self, runner, **kwargs):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    @property
    def experiment(self):
        return self.wandb

    @torch.no_grad()
    def add_image(self, name, img_rgb, data_sample, iter, **kwargs):
        if data_sample is not None:
            # table 总是无法显示，暂时不使用
            # if self.image_table is None:
            # self.data_at = self.wandb.Artifact("GT-PRED", type="gt-pred")
            # image_table = self.wandb.Table(columns=["mode", 'gt', 'pred'])
            # 必须要有 class,目前写死数据
            class_set = self.wandb.Classes([{'id': id, 'name': name} for id, name in [(0, 'ok'), (1, 'ng')]])

            out_data = [name + str(iter)]
            if 'gt_instances' in data_sample:
                gt_instances = data_sample.gt_instances.to('cpu')
                box_data = gt_instances.bboxes.tensor.tolist()
                box_labels = gt_instances.labels.tolist()
                bbox_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                              "class_id": int(label),
                              "domain": "pixel"} for (xyxy, label) in zip(box_data, box_labels)]
                boxes = {"gt": {"box_data": bbox_data, "class_labels": {0: 'ok', 1: 'ng'}}}
                gt_data = self.wandb.Image(img_rgb, boxes=boxes, classes=class_set)
                out_data.append(gt_data)
            else:
                out_data.append(self.wandb.Image(img_rgb))

            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances.to('cpu')
                box_data = pred_instances.bboxes.tensor.tolist()
                box_labels = pred_instances.labels.tolist()
                box_scores = pred_instances.scores.tolist()
                bbox_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                              "class_id": int(label),
                              "scores": {"class_score": score},
                              "domain": "pixel"} for (xyxy, score, label) in zip(box_data, box_scores, box_labels)]
                boxes = {"pred": {"box_data": bbox_data, "class_labels": {0: 'ok', 1: 'ng'}}}
                pred_data = self.wandb.Image(img_rgb, boxes=boxes, classes=class_set)
                out_data.append(pred_data)
            else:
                out_data.append(self.wandb.Image(img_rgb))

            # image_table.add_data(out_data[0], out_data[1], out_data[2])
            # self.wandb.log({'image': image_table}, commit=self.commit)
            self.wandb.log({name: [gt_data,pred_data]}, commit=self.commit)

            # self.data_at.add(self.image_table, "GT-PRED")
            # self.wandb.run.use_artifact(self.data_at)

    def close(self):
        if hasattr(self, "wandb"):  # doesn't exist when the code fails at import
            self.wandb.join()
