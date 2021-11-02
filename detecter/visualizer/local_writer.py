import os

import torch

import cvcore
import numpy as np

from .base_writer import BaseWriter
from .visualizer import DetVisualizer
from .builder import WRITERS

__all__ = ['LocalWriter']


@WRITERS.register_module()
class LocalWriter(BaseWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, work_dir=None, default_name='runtime_data', show=False, **kwargs):
        self.work_dir = work_dir
        self.kwargs = kwargs
        self.default_name = default_name
        self._last_write = 1
        self.visualizer = None
        self.show = show

    @property
    def experiment(self):
        return self

    def init(self, runner, **kwargs):
        if self.work_dir is None:
            self.work_dir = os.path.join(runner.work_dir, self.default_name)
        cvcore.mkdir_or_exist(self.work_dir)

        self.visualizer = DetVisualizer()

    # 原则上 data_sample 里面也有每张图片相关的meta信息，例如 filename，可以用于此处文件保存
    @torch.no_grad()
    def add_image(self, name, img_rgb, data_sample, iter, **kwargs):
        self.visualizer.set_image(img_rgb)
        if data_sample is not None:
            # TODO: 更好的判断内部是否有 gt_ 或者 pred_ 对象
            gt_vis_img = None
            pred_vis_img = None
            if 'gt_instances' in data_sample:
                self.visualizer.draw(data_sample, show_gt=True, show_pred=False)
                gt_vis_img = self.visualizer.get_image()
            if 'pred_instances' in data_sample:
                self.visualizer.draw(data_sample, show_gt=False, show_pred=True)
                pred_vis_img = self.visualizer.get_image()

            if gt_vis_img is not None and pred_vis_img is not None:
                concat = np.concatenate((gt_vis_img, pred_vis_img), axis=1)
                if self.show:
                    self.visualizer.show(concat, str(iter) + f'_{name}')
                else:
                    img_name = os.path.join(self.work_dir, str(iter) + f'_{name}.jpg')
                    self.visualizer.save(img_name, concat)
