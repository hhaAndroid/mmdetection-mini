from ..utils import Hook, HOOKS
from collections import Counter
from . import Logger

__all__ = ['DefaultLoggerHook', 'get_best_param_group_id']


def get_best_param_group_id(optimizer):
    # NOTE: some heuristics on what LR to summarize
    # summarize the param group with most parameters
    largest_group = max(len(g["params"]) for g in optimizer.param_groups)

    if largest_group == 1:
        # If all groups have one parameter,
        # then find the most common initial LR, and use it for summary
        lr_count = Counter([g["lr"] for g in optimizer.param_groups])
        lr = lr_count.most_common()[0][0]
        for i, g in enumerate(optimizer.param_groups):
            if g["lr"] == lr:
                return i
    else:
        for i, g in enumerate(optimizer.param_groups):
            if len(g["params"]) == largest_group:
                return i


@HOOKS.register_module()
class DefaultLoggerHook(Hook):
    def __init__(self, interval=1):
        self.interval = interval
        self._append_funcs = None

    def parse_log(self, log_storage_value):
        log_items = []
        # TODO： support loss smooth print
        for values in log_storage_value:
            for name, val in values.items():
                if isinstance(val, float):
                    val = f'{val:.4f}'
                log_items.append(f'{name}: {val}')
        return ', '.join(log_items)

    def append_runner_type(self, runner):
        if runner.runner_type == 'iter':
            runner.log_storage.insert(0, {'Iter': f'[{runner.iter}][{runner.max_iters}]'})
        else:
            runner.log_storage.insert(0, {'Epoch': f'[{runner.epoch}][{runner.iter}/{runner.max_iters}]'})

    def append_lr(self, runner):
        _best_param_group_id = get_best_param_group_id(runner.optimizer)
        lr = runner.optimizer.param_groups[_best_param_group_id]["lr"]

        runner.log_storage.append({'lr': lr})

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            # TODO: 自动找到所有 append 开头的函数自动执行
            self.append_lr(runner)
            self.append_runner_type(runner)

            log_str = self.parse_log(runner.log_storage.values())
            Logger.info(log_str)
