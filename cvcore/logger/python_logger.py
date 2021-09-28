from .base_logger import BaseLogger
from .builder import LOGGERS
import logging
from ..utils import dist_comm
from termcolor import colored
import sys

__all__ = ['PyLogging']

# DEFAULT_LOG_FORMAT = '%(asctime)s%(message)s'
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class _ColorfulFormatter(logging.Formatter):

    def __init__(self, log_format):
        super(_ColorfulFormatter, self).__init__(log_format)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            color_log = colored("WARNING:" + log, "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR:
            color_log = colored("ERROR:" + log, "red", attrs=["blink"])
        elif record.levelno == logging.CRITICAL:
            color_log = colored("CRITICAL:" + log, "red", attrs=["blink"])
        elif record.levelno == logging.INFO:
            color_log = colored("INFO:" + log, "green", attrs=["blink"])
        else:
            return log
        return color_log


DISP_FORMATTER = {
    'color': _ColorfulFormatter,
    'normal': logging.Formatter
}

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


@LOGGERS.register_module()
class PyLogging(BaseLogger):
    def __init__(self, name='det', log_file=None, log_level='info', log_format=DEFAULT_LOG_FORMAT, disp_format='color',
                 file_mode='w'):
        assert disp_format in DISP_FORMATTER
        assert str(log_level).lower() in LOG_LEVEL_DICT, 'only support {}'.format(LOG_LEVEL_DICT.keys())
        log_level = LOG_LEVEL_DICT[str(log_level).lower()]

        self.logger = logging.getLogger(name)

        rank = dist_comm.get_rank()
        log_formatter = logging.Formatter(log_format)
        # only rank 0 will add a FileHandler
        if rank == 0 and log_file is not None:
            file_handler = logging.FileHandler(log_file, file_mode)
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)

        # disp
        disp_stream_handler = logging.StreamHandler(stream=sys.stdout)
        disp_formatter = DISP_FORMATTER[disp_format](log_format)
        disp_stream_handler.setFormatter(disp_formatter)
        disp_stream_handler.setLevel(log_level)
        self.logger.addHandler(disp_stream_handler)

        if rank == 0:
            self.logger.setLevel(log_level)
        else:
            self.logger.setLevel(logging.ERROR)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg)

    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg)
