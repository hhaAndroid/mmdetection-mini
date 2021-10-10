from cvcore import Registry, build_from_cfg
from .base_logger import BaseLogger

__all__ = ['LOGGERS', 'Logger']

LOGGERS = Registry('loggers')


def check_logger(func):
    def run(cls, msg, *args, **kwargs):
        if cls.logger is None:
            raise Exception('Logger is None, you should Logger.init() first!')
        result = func(cls, msg, *args, **kwargs)
        return result

    return run


class Logger:
    logger = None

    @staticmethod
    def init(backend=None):
        if Logger.logger is None:
            Logger._setup_logger(backend)
        return Logger

    @staticmethod
    def _setup_logger(backend):
        if backend is None:
            backend = dict(type='PyLogging')
        logger = build_from_cfg(backend, LOGGERS)
        assert isinstance(logger, BaseLogger)
        Logger.logger = logger

    @classmethod
    @check_logger
    def debug(cls, msg, *args, **kwargs):
        Logger.logger.debug(msg)

    @classmethod
    @check_logger
    def info(cls, msg, *args, **kwargs):
        Logger.logger.info(msg)

    @classmethod
    @check_logger
    def warning(cls, msg, *args, **kwargs):
        Logger.logger.warning(msg)

    @classmethod
    @check_logger
    def warn(cls, msg, *args, **kwargs):
        Logger.logger.warn(msg)

    @classmethod
    @check_logger
    def error(cls, msg, *args, **kwargs):
        Logger.logger.error(msg)



