# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import cvcore
from cvcore.utils import dist_comm
from torch.nn.parallel import DistributedDataParallel
from cvcore import Hook, get_priority,Logger

import copy
import time
import datetime
import torch
from cvcore import build_from_cfg, HOOKS, EventStorage
from contextlib import ExitStack, contextmanager
import torch
from detecter.evaluation import DatasetEvaluator,inference_context,EVALUATORS
from detecter.visualizer import PeriodicWriterHook
import torch.nn as nn


__all__ = ['SimpleTestRunner']


def create_ddp_model(model, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if dist_comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [dist_comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    return ddp


class SimpleTestRunner:
    def __init__(self,  model,logger=None,work_dir=None,cfg=None):
        self.cfg = cfg
        self.model = create_ddp_model(model, broadcast_buffers=False)

        # create work_dir
        if cvcore.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            cvcore.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # dump config
        # cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

        logger_cfg = cfg.logger
        if 'log_file' not in logger_cfg:
                logger_cfg['log_file'] = log_file
        logger = Logger.init(logger_cfg)
        self.logger = logger

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = cvcore.get_rank(), cvcore.get_world_size()
        self.timestamp = cvcore.get_time_str()

        self._iter = 0

        self.log_storage = None
        self.event_storage = None
        self.runner_type = 'iter'
        self.evaluator = None
        self._hooks=[]

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg):
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Notes:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = cvcore.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def run(self,dataloader):
        cp_evaluator_cfg = copy.deepcopy(self.cfg.evaluator)
        cp_evaluator_cfg['dataloader'] = dataloader

        evaluator = build_from_cfg(cp_evaluator_cfg, EVALUATORS)
        assert isinstance(evaluator, DatasetEvaluator)

        self.logger.info("Start inference on {} batches".format(len(dataloader)))

        with EventStorage(0) as self.event_storage:
            self.call_hook('before_run')
            results=self.inference_on_dataset(dataloader,evaluator)
            return results


    def inference_on_dataset(self,dataloader,evaluator):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            The return value of `evaluator.evaluate()`
        """
        num_devices = dist_comm.get_world_size()
        self.logger.info("Start inference on {} batches".format(len(dataloader)))

        total = len(dataloader)  # inference data loader must have a fixed length
        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        eval_log_interv = 50
        self.val_iter=0

        with ExitStack() as stack:
            if isinstance(self.model, nn.Module):
                stack.enter_context(inference_context(self.model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(dataloader):
                self.call_hook('before_iter')
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = self.model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                if idx % eval_log_interv == 0:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    self.logger.info(
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    )
                start_data_time = time.perf_counter()

                self.event_storage.iter +=1
                self.val_iter +=1

                self.call_hook('after_iter')


        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        self.logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        self.logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results





