from .builder import LR_SCHEDULERS, LR_PARAM_SCHEDULERS
from cvcore.utils import Hook, build_from_cfg

__all__ = ['DefaultLrScheduler']


@LR_SCHEDULERS.register_module()
class DefaultLrScheduler(Hook):
    def __init__(self, optimizer, warmup_param_scheduler, regular_param_scheduler, by_epoch=True, warmup_by_epoch=False,
                 warmup_iter_or_epochs=0):
        self.optimizer = optimizer
        self.warmup_param_scheduler = build_from_cfg(warmup_param_scheduler, LR_PARAM_SCHEDULERS)
        self.regular_param_scheduler = build_from_cfg(regular_param_scheduler, LR_PARAM_SCHEDULERS)
        self.by_epoch = by_epoch
        self.warmup_by_epoch = warmup_by_epoch
        self.warmup_iter_or_epochs = warmup_iter_or_epochs

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
        if self.warmup_by_epoch and cur_epoch < self.warmup_iters:
            lr = self.warmup_param_scheduler()
            self._set_lr(runner, lr)
            return

        if self.by_epoch:
            regular_lr = self.regular_param_scheduler(runner)
            self._set_lr(runner, regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.warmup_by_epoch and cur_iter < self.warmup_iters:
            lr = self.warmup_param_scheduler()
            self._set_lr(runner, lr)
            return

        if not self.by_epoch:
            regular_lr = self.regular_param_scheduler(runner)
            self._set_lr(runner, regular_lr)
