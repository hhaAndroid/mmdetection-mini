from .builder import LR_SCHEDULERS, LR_PARAM_SCHEDULERS
from cvcore.utils import Hook, build_from_cfg
import copy
from .param_scheduler import ParamScheduler

__all__ = ['DefaultLrScheduler']


@LR_SCHEDULERS.register_module()
class DefaultLrScheduler(Hook):
    def __init__(self, optimizer, warmup_param_scheduler, regular_param_scheduler, by_epoch=True, warmup_by_epoch=False,
                 warmup_iter_or_epochs=0):
        self.optimizer = optimizer
        cp_warmup_param_scheduler = copy.deepcopy(warmup_param_scheduler)
        cp_warmup_param_scheduler['by_epoch'] = warmup_by_epoch
        self.warmup_param_scheduler = build_from_cfg(cp_warmup_param_scheduler, LR_PARAM_SCHEDULERS)
        assert isinstance(self.warmup_param_scheduler, ParamScheduler)

        cp_regular_param_scheduler = copy.deepcopy(regular_param_scheduler)
        cp_regular_param_scheduler['by_epoch'] = by_epoch
        self.regular_param_scheduler = build_from_cfg(cp_regular_param_scheduler, LR_PARAM_SCHEDULERS)

        self.by_epoch = by_epoch
        self.warmup_by_epoch = warmup_by_epoch
        self.warmup_iter_or_epochs = warmup_iter_or_epochs

        self.base_lr = []  # initial lr for all param groups

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in runner.optimizer.param_groups
            ]

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def before_train_epoch(self, runner):
        cur_epoch = runner.epoch
        if self.warmup_by_epoch and cur_epoch < self.warmup_iter_or_epochs:
            lr_groups = [self.warmup_param_scheduler(runner, base_lr) for base_lr in self.base_lr]
            self._set_lr(runner, lr_groups)
            return

        if self.by_epoch:
            lr_groups = [self.regular_param_scheduler(runner, base_lr) for base_lr in self.base_lr]
            self._set_lr(runner, lr_groups)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.warmup_by_epoch and cur_iter < self.warmup_iter_or_epochs:
            lr_groups = [self.warmup_param_scheduler(runner, base_lr) for base_lr in self.base_lr]
            self._set_lr(runner, lr_groups)
            return

        if not self.by_epoch:
            lr_groups = [self.regular_param_scheduler(runner, base_lr) for base_lr in self.base_lr]
            self._set_lr(runner, lr_groups)
