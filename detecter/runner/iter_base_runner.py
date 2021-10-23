# Copyright (c) Open-MMLab. All rights reserved.
import time
from .base_runner import BaseRunner
from .builder import RUNNERS
from cvcore.utils import EventStorage, LoggerStorage
import cvcore
from cvcore import Logger

__all__ = ['IterBasedRunner']


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def train_step(self, **kwargs):
        assert self.model.training
        data_batch = next(self.dataloader)
        self.call_hook('before_train_iter')

        losses = self.model(data_batch, **kwargs)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        self.scheduler.step(self)

        self.call_hook('after_train_iter')
        self._iter += 1

    def run(self, **kwargs):
        """Start running.
        """

        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         cvcore.get_host_info(), work_dir)
        # self.logger.info('Hooks will be executed in the following order:\n%s',
        #                  self.get_hook_info())

        self.dataloader = iter(self.dataloader)

        with EventStorage(self.iter) as self.event_storage:
            with LoggerStorage() as self.log_storage:
                self.call_hook('before_run')

                while self.iter < self._max_iters:
                    self.train_step(**kwargs)

                    self.log_storage.clear()

                time.sleep(1)  # wait for some hooks like loggers to finish
                self.call_hook('after_run')
