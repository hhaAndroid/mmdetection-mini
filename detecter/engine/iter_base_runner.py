# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
from cvcore.utils import EventStorage

__all__ = ['IterBasedRunner']


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train_step(self, **kwargs):
        assert self.model.training
        data_batch = next(self.data_loader)
        self.call_hook('before_train_iter')

        self.scheduler.step()
        outputs = self.model(data_batch, **kwargs)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        if not isinstance(outputs, dict):
            raise TypeError('model() must return a dict')
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._iter += 1

    def run(self, **kwargs):
        """Start running.
        """
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())

        with EventStorage(self.iter) as self.storage:
            self.call_hook('before_run')

            while self.iter < self._max_iters:
                self.train_step(**kwargs)

            time.sleep(1)  # wait for some hooks like loggers to finish
            self.call_hook('after_run')
