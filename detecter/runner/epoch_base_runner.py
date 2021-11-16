# Copyright (c) Open-MMLab. All rights reserved.
import time
import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from cvcore.utils import LoggerStorage
from detecter.visualizer import EventWriterStorage
import cvcore
from contextlib import ExitStack
from cvcore import build_from_cfg, Logger
from detecter.evaluation.base_evaluator import DatasetEvaluator
from detecter.evaluation import EVALUATORS, inference_context, print_csv_format
from cvcore.utils import dist_comm
import copy
import torch
import torch.nn as nn
from detecter.visualizer.builder import WRITERS
from detecter.visualizer.base_writer import BaseWriter
from detecter.others import build_func_storage, DefaultFuncStorage

__all__ = ['EpochBasedRunner']


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run(self, data_loaders, workflow, **kwargs):
        self.runner_type = 'epoch'

        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        self.logger.warn(f'Start running, host: {cvcore.get_host_info()}, work_dir: {self.work_dir}')
        # self.logger.info('Hooks will be executed in the following order:\n%s',
        #                  self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)

        # build writer
        writers = self.cfg.get('writer', [])
        if not isinstance(writers, list):
            writers = [writers]

        writers_obj = []
        for w in writers:
            if isinstance(w, dict):
                w = build_from_cfg(w, WRITERS)
            else:
                assert isinstance(w, BaseWriter), w
            w.init(self)
            writers_obj.append(w)

        if len(data_loaders) == 2:
            self.val_event_storage = EventWriterStorage(writers_obj)

        runtime_func = self.cfg.get('runtime_func', None)
        if runtime_func:
            runtime_func = build_func_storage(runtime_func)
            assert isinstance(runtime_func, DefaultFuncStorage)

        with EventWriterStorage(writers_obj, self.iter + 1) as self.event_storage:
            with LoggerStorage() as self.log_storage:
                if runtime_func:
                    with runtime_func:
                        self._run(workflow, data_loaders, **kwargs)
                else:
                    self._run(workflow, data_loaders, **kwargs)

    def _run(self, workflow, data_loaders, **kwargs):
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.data_loader = data_loader

        # 避免每个epoch输出的随机性固定
        if hasattr(self.data_loader.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            self.data_loader.sampler.set_epoch(self._epoch)
        elif hasattr(self.data_loader.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            self.data_loader.batch_sampler.sampler.set_epoch(self._epoch)

        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')

            losses = self.model(data_batch, **kwargs)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.scheduler.step(self)

            self.call_hook('after_train_iter')
            self._iter += 1
            self.event_storage.iter = self._iter

            self.log_storage.clear()

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, dataloader, **kwargs):
        self.dataloader = dataloader
        cp_evaluator_cfg = copy.deepcopy(self.cfg.evaluator)
        cp_evaluator_cfg['dataloader'] = dataloader

        evaluator = build_from_cfg(cp_evaluator_cfg, EVALUATORS)
        assert isinstance(evaluator, DatasetEvaluator)

        Logger.info("Start inference on {} batches".format(len(dataloader)))

        total = len(dataloader)  # inference data loader must have a fixed length
        evaluator.reset()

        self.val_iter = 0

        self.call_hook('before_val_epoch')
        with self.val_event_storage:
            with ExitStack() as stack:
                if isinstance(self.model, nn.Module):
                    stack.enter_context(inference_context(self.model))
                stack.enter_context(torch.no_grad())

                for idx, inputs in enumerate(dataloader):
                    self.call_hook('before_val_iter')
                    outputs = self.model(inputs)

                    self.val_event_storage.iter += 1

                    self.call_hook('after_val_iter')
                    evaluator.process(inputs, outputs)
                    self.val_iter += 1

                    self.log_storage.clear()

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
