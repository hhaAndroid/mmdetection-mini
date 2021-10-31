from ..utils import EventWriter
from cvcore.utils import get_event_storage
from .builder import WRITERS

__all__ = ['WandbWriter']


@WRITERS.register_module()
class WandbWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self,init_kwargs=None,commit=True,with_step=True, **kwargs):
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
        self._last_write = -1
        self._writer = None
        self._window_size=20

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb


    def init(self, runner):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()


    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self.wandb.log((k,v), step=iter, commit=self.commit)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self.wandb.log({img_name: self.wandb.Image(img.transpose(1,2,0))},step=step_num)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            # for params in storage._histograms:
            #     self._writer.add_histogram_raw(**params)
            storage.clear_histograms()


    def close(self):
        self.wandb.join()

