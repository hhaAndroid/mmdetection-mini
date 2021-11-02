from .base_writer import BaseWriter
from .builder import WRITERS
import torch

__all__ = ['TensorboardWriter']


@WRITERS.register_module()
class TensorboardWriter(BaseWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, work_dir=None, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self.work_dir = work_dir
        self.kwargs = kwargs
        self._last_write = -1
        self._writer = None

    def init(self, runner, **kwargs):
        if self.work_dir is None:
            self.work_dir = runner.work_dir

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(self.work_dir, **self.kwargs)

    @property
    def experiment(self):
        return self._writer

    @torch.no_grad()
    def add_image(self, name, img_rgb, data_sample, iter, **kwargs):
        if data_sample is not None:
            # TODO: 还是用 detvisualizer 更好靠谱一些，tensorboard 功能太弱了
            # 无法定制颜色， 这api感觉比较搞笑
            if 'gt_instances' in data_sample:
                self._writer.add_image_with_boxes('gt_'+name, img_rgb, data_sample.gt_instances.bboxes.tensor.cpu(), iter,
                                                  dataformats='HWC')
            if 'pred_instances' in data_sample:
                self._writer.add_image_with_boxes('pred_' + name, img_rgb, data_sample.pred_instances.bboxes.tensor.cpu(),
                                                  iter,
                                                  dataformats='HWC')

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()
