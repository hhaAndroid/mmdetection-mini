# 这里默认都是只考虑主进程，是否要同时考虑多进程情况下的聚合操作？
from abc import ABC, abstractmethod

__all__ = ['BaseWriter']


class BaseWriter(ABC):
    """Base class for experiment writer.
    """

    def init(self, runner, **kwargs):
        pass

    @property
    @abstractmethod
    def experiment(self):
        """Return the experiment object associated with this writer."""

    def add_hyperparams(self, name, hyperparams, iter=0,  **kwargs):
        """Record hyperparameters.
        """
        pass

    def add_graph(self, model, input_array=None, iter=0):
        """Record model graph.
        """
        pass

    def add_scalar(self, name, value, iter,  **kwargs) -> None:
        pass

    def add_image(self, name, img_tensor, data_sample, iter, **kwargs):
        pass

    def add_data(self, name, data, iter,  **kwargs):
        pass

