# Copyright (c) Open-MMLab. All rights reserved.
import time
from .base_runner import BaseRunner
from .builder import RUNNERS
from cvcore.utils import EventStorage, LoggerStorage
import cvcore
from contextlib import ExitStack, contextmanager
from cvcore import build_from_cfg, Logger, get_event_storage
from detecter.evaluation.base_evaluator import DatasetEvaluator
from detecter.evaluation import EVALUATORS,inference_context,print_csv_format
from cvcore.utils import dist_comm
import copy
import torch
import torch.nn as nn


__all__ = ['IterBasedRunner']



@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def train(self, dataloader, **kwargs):
        assert self.model.training
        self.dataloader=dataloader
        data_batch = next(dataloader)
        self.call_hook('before_train_iter')

        losses = self.model(data_batch, **kwargs)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        self.scheduler.step(self)

        self.call_hook('after_train_iter')
        self._iter += 1

        self.event_storage.iter = self._iter


    def val(self, dataloader, **kwargs):
        self.dataloader = dataloader
        cp_evaluator_cfg = copy.deepcopy(self.cfg.evaluator)
        cp_evaluator_cfg['dataloader'] = dataloader

        evaluator = build_from_cfg(cp_evaluator_cfg, EVALUATORS)
        assert isinstance(evaluator, DatasetEvaluator)

        Logger.info("Start inference on {} batches".format(len(dataloader)))

        total = len(dataloader)  # inference data loader must have a fixed length
        evaluator.reset()

        self.val_iter=0

        with EventStorage() as val_event_storage:
            self.call_hook('before_val_epoch')
            with ExitStack() as stack:
                if isinstance(self.model, nn.Module):
                    stack.enter_context(inference_context(self.model))
                stack.enter_context(torch.no_grad())

                for idx, inputs in enumerate(dataloader):
                    self.call_hook('before_val_iter')
                    outputs = self.model(inputs)

                    val_event_storage.iter +=1

                    self.call_hook('after_val_iter')
                    evaluator.process(inputs, outputs)
                    self.val_iter +=1

            results = evaluator.evaluate()
            # An evaluator may return None when not in main process.
            # Replace it by an empty dict instead to make it easier for downstream code to handle
            if results is None:
                results = {}

            if dist_comm.is_main_process():
                assert isinstance(
                    results, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results
                )
                # Logger.info("Evaluation results for {} in csv format:".format(val_dataset.__name__))
                print_csv_format(results)
            self.call_hook('after_val_epoch')


    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.
        """
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         cvcore.get_host_info(), work_dir)
        # self.logger.info('Hooks will be executed in the following order:\n%s',
        #                  self.get_hook_info())

        iter_loaders = [iter(data_loaders[0]),data_loaders[1]]

        with EventStorage(self.iter) as self.event_storage:

            with LoggerStorage() as self.log_storage:
                self.call_hook('before_run')

                while self.iter < self._max_iters:
                    for i, flow in enumerate(workflow):
                        mode, iters = flow

                        iter_runner = getattr(self, mode)
                        for _ in range(iters):
                            if mode == 'train' and self.iter >= self._max_iters:
                                break
                            iter_runner(iter_loaders[i], **kwargs)
                            self.log_storage.clear()

                time.sleep(1)  # wait for some hooks like loggers to finish
                self.call_hook('after_run')
