from .builder import LOGGERS
from .base_logger import BaseLogger
from ..utils import dist_comm
import sys
import inspect
from loguru import logger

__all__ = ['LoguruLogging']


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


@LOGGERS.register_module()
class LoguruLogging(BaseLogger):
    def __init__(self, name='det', log_file=None, log_level='info', file_mode='w'):
        from loguru import logger
        self.logger = logger
        # TODO： refine
        loguru_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

        rank = dist_comm.get_rank()

        if rank == 0 and log_file is not None:
            logger.add(
                sys.stderr,
                format=loguru_format,
                level="INFO",
                enqueue=True,
            )
            logger.add(log_file)

        # redirect_sys_output("INFO")

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg)

    def warn(self, msg, *args, **kwargs):
        self.logger.warning(msg)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg)
