from contextlib import contextmanager

from cvcore import Registry, build_from_cfg
import torch
from .base_writer import BaseWriter

__all__ = ['WRITERS', 'VISUALIZERS', 'EventWriterStorage', 'get_event_storage']

WRITERS = Registry('writers')
VISUALIZERS = Registry('visualizers')

_CURRENT_EVENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _CURRENT_EVENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_EVENT_STORAGE_STACK[-1]


class EventWriterStorage:
    """
    The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, writers, start_iter=1):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

        if not isinstance(writers, list):
            writers = [writers]

        writers_obj = []
        for w in writers:
            if isinstance(w, dict):
                w = build_from_cfg(w, WRITERS)
            else:
                assert isinstance(w, BaseWriter), w
            writers_obj.append(w)

        self._writers = writers_obj

    @property
    def experiments(self):
        return [writer.experiment for writer in self._writers]

    def add_image(self, img_name, img_rgb, data_sample):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.

        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        for w in self._writers:
            w.add_image(img_name, img_rgb, data_sample, self._iter)

    def add_scalar(self, name, value):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """

        name = self._current_prefix + name
        value = float(value)

        for w in self._writers:
            w.add_scalar(name, value, self._iter)

    def add_data(self, name, data):
        for w in self._writers:
            w.add_scalar(name, data, self._iter)

    def add_histogram(self, hist_name, hist_tensor, bins=1000):
        """
        Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()

        # Create a histogram with PyTorch
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = dict(
            tag=hist_name,
            min=ht_min,
            max=ht_max,
            num=len(hist_tensor),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor ** 2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self._iter,
        )
        for w in self._writers:
            w.add_histogram(hist_name, hist_params, self._iter)

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_EVENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_EVENT_STORAGE_STACK[-1] == self
        _CURRENT_EVENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix
