# Copyright (c) Open-MMLab. All rights reserved.
import copy
import datetime
from detecter.visualizer import EventWriterStorage
from contextlib import ExitStack
import torch.nn as nn
import time
import os.path as osp
import torch

import cvcore
from cvcore import build_from_cfg, HOOKS, Logger
from cvcore.utils import dist_comm

from ..model import build_detector
from ..dataset import build_dataset
from ..dataloader import build_dataloader
from ..utils import auto_replace_data_root, wrapper_model
from ..utils.checkpoint import DetectionCheckpointer
from ..evaluation import DatasetEvaluator, inference_context, EVALUATORS

from cvcore import Hook, get_priority
from ..visualizer import WRITERS, BaseWriter


__all__ = ['DefaultTester']


class DefaultTester:
    def __init__(self, cfg):
        self.logger = None
        # The order is more critical
        self.cfg = auto_replace_data_root(cfg)
        self.setup_cfg()
        self.setup_logger()

        self.detector = wrapper_model(self.build_detector())
        self.test_dataset = self.build_test_dataset()
        self.test_dataloader = self.build_test_dataloader()

        self._rank, self._world_size = cvcore.get_rank(), cvcore.get_world_size()
        self.timestamp = cvcore.get_time_str()

        self.event_storage = None
        self.evaluator = None
        self._hooks = []

    def setup_cfg(self):
        # import modules from string list.
        if self.cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**self.cfg['custom_imports'])
            # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if self.cfg.get('work_dir', None) is not None:
            # use config filename as default work_dir if cfg.work_dir is None
            self.cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(self.cfg.config))[0])
        else:
            self.cfg.work_dir = None

    def setup_logger(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.cfg.work_dir, f'{timestamp}.log')
        logger_cfg = self.cfg.logger
        if 'log_file' not in logger_cfg:
            logger_cfg['log_file'] = log_file
        self.logger = Logger.init(logger_cfg)

    def build_detector(self, detector=None, skip_ckpt=False):
        if detector is not None:
            if not skip_ckpt:
                checkpointer = DetectionCheckpointer(detector)
                checkpointer.load(self.cfg.checkpoint)
            return detector

        self.cfg.model.pretrained = None
        self.cfg.model.train_cfg = None

        if 'vis_interval' in self.cfg:
            detector = build_detector(self.cfg.model, dict(vis_interval=self.cfg.vis_interval))
        else:
            detector = build_detector(self.cfg.model)

        checkpointer = DetectionCheckpointer(detector)
        checkpointer.load(self.cfg.checkpoint)
        return detector

    def build_test_dataset(self, dataset=None):
        if dataset is not None:
            return dataset
        return build_dataset(self.cfg.data.test)

    def build_test_dataloader(self, dataloader=None):
        if dataloader is not None:
            return dataloader
        return build_dataloader(self.cfg.dataloader.test, self.test_dataset)

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

    def run(self):
        cp_evaluator_cfg = copy.deepcopy(self.cfg.evaluator)
        cp_evaluator_cfg['dataloader'] = self.test_dataloader

        evaluator = build_from_cfg(cp_evaluator_cfg, EVALUATORS)
        assert isinstance(evaluator, DatasetEvaluator)

        self.logger.info("Start inference on {} batches".format(len(self.test_dataloader)))

        writers = self.cfg.writer
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

        with EventWriterStorage(writers_obj, 0) as self.event_storage:
            self.call_hook('before_run')
            results = self.inference_on_dataset(self.test_dataloader, evaluator)
            return results

    def inference_on_dataset(self, dataloader, evaluator):
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
        self.val_iter = 0

        with ExitStack() as stack:
            if isinstance(self.detector, nn.Module):
                stack.enter_context(inference_context(self.detector))
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
                outputs = self.detector(inputs)
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

                self.event_storage.iter += 1
                self.val_iter += 1

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

        self.logger.info(results)
        return results




