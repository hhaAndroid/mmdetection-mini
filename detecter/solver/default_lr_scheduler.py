
from .builder import LR_SCHEDULERS

__all__=['DefaultLrScheduler']


@LR_SCHEDULERS.register_module()
class DefaultLrScheduler(Hook):
    def __init__(self,warmup_param_scheduler,regular_param_scheduler,by_epoch=True,warmup_by_epoch=False,warmup_iter_or_epochs=0):
        pass

    
    def before_train_epoch(self, runner):
        cur_epoch = runner.epoch
        if self.warmup_by_epoch and cur_epoch < self.warmup_iters:
            lr=self.warmup_param_scheduler()
            self._set_lr(runner, lr)
            return

        if self.by_epoch:
            regular_lr = self.regular_param_scheduler(runner)
            self._set_lr(runner, regular_lr)


    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.warmup_by_epoch and cur_iter < self.warmup_iters:
            lr=self.warmup_param_scheduler()
            self._set_lr(runner, lr)
            return

        if not self.by_epoch:
            regular_lr = self.regular_param_scheduler(runner)
            self._set_lr(runner, regular_lr)
