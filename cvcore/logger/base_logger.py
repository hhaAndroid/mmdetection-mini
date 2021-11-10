from abc import ABCMeta, abstractmethod

__all__ = ['BaseLogger']


class BaseLogger(metaclass=ABCMeta):

    @abstractmethod
    def debug(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def info(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def warning(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def warn(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def critical(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def error(self, msg, *args, **kwargs):
        pass

