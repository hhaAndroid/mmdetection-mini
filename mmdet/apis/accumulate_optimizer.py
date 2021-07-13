from mmcv.runner.hooks import HOOKS, Hook
from torch.nn.utils import clip_grad
import numpy as np


@HOOKS.register_module()
class AccumulateOptimizerHook(Hook):

    def __init__(self,
                 grad_clip=None,
                 update_iterval=1,
                 warm_iter=1000):
        self.grad_clip = grad_clip
        self.update_iterval = update_iterval
        # self.warm_iter = warm_iter
        self.warmup_epochs = 3

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        warmup_total_iters = max(round(self.warmup_epochs * len(runner.data_loader)), 1000)
        accumulate = max(1, np.interp(runner.iter, [0, warmup_total_iters], [1, self.update_iterval]).round())
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        if runner.iter % accumulate == 0:
            runner.optimizer.step()
            runner.optimizer.zero_grad()



