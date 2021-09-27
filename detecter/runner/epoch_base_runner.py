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
# from .checkpoint import save_checkpoint
# from .utils import get_host_info
from cvcore.utils import EventStorage


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train_step(self, data_batch, **kwargs):
        assert self.model.training
        self.scheduler.step()
        outputs = self.model(data_batch, **kwargs)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self.outputs = outputs

    def train_epoch(self, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.train_step(data_batch, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self, **kwargs):
        """Start running.
        """
        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')
        self._max_iters = self._max_epochs * len(self.data_loader)
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())

        with EventStorage(self.epoch) as self.storage:
            self.call_hook('before_run')

            while self.epoch < self._max_epochs:
                self.train_epoch(**kwargs)

            time.sleep(1)  # wait for some hooks like loggers to finish
            self.call_hook('after_run')

