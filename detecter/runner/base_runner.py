# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp

from abc import ABCMeta, abstractmethod

import cvcore
# from ..parallel import is_module_wrapper
from cvcore import HOOKS, Hook, get_priority

# from .log_buffer import LogBuffer
from ..evaluation import build_evaluator as build_default_evaluator
from ..evaluation import EvalHook,eval_func

__all__ = ['BaseRunner']


class BaseRunner(metaclass=ABCMeta):

    def __init__(self,
                 model,
                 dataloader,
                 optimizer,
                 scheduler,
                 logger,
                 meta=None,
                 work_dir=None,
                 max_iters=None,
                 max_epochs=None,
                 cfg=None):

        self.cfg=cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler  # hook
        assert isinstance(self.scheduler, Hook)

        self.logger = logger
        self.meta = meta
        # create work_dir
        if cvcore.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            cvcore.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = cvcore.get_rank(), cvcore.get_world_size()
        self.timestamp = cvcore.get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters

        self.log_storage = None
        self.event_storage = None
        self.runner_type = 'iter'
        self.evaluator = None

        self.register_hook(self.scheduler, priority=99)

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def run(self):
        pass

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

    def register_evaluator_hook(self, evaluator_cfg, priority=100):
        evaluator_cfg = evaluator_cfg.copy()
        if 'priority' in evaluator_cfg:
            priority=evaluator_cfg['priority']

        def test_and_save_results():
            self._last_eval_results = eval_func(self.cfg, self.model)
            return self._last_eval_results

        by_epoch=evaluator_cfg.get('by_epoch',True)
        eval_period=evaluator_cfg.get('eval_period',1)

        self.register_hook(EvalHook(test_and_save_results,by_epoch,eval_period),priority)




