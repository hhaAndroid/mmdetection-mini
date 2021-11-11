from ..utils import Hook, HOOKS
import time
import datetime
import torch
import torch.distributed as dist
from collections import Counter
from . import Logger
from ..utils import AvgBuffer

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


def _get_max_memory(runner):
    device = runner.model.device
    if device == 'cpu':
        return 0
    # 打印从程序开始跑后的峰值内存，不可能会下降
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([mem / (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    if runner.world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()


@HOOKS.register_module()
class DefaultLoggerHook(Hook):
    def __init__(self, interval=1):
        self.interval = interval
        self._append_funcs = None
        self._avg_buffer = AvgBuffer()
        self.time_sec_tot = 0

    def before_run(self, runner):
        self.t = time.time()
        self.start_iter = runner.iter

    def before_train_iter(self, runner):
        self._avg_buffer.update({'data_time': time.time() - self.t})

    def append_time(self, runner):
        self._avg_buffer.average(self.interval)
        runner.log_storage.append({'data_time': self._avg_buffer.output['data_time']})
        runner.log_storage.append({'time': self._avg_buffer.output['time']})

        # 到目前为止。总共跑了多少时间
        self.time_sec_tot += (self._avg_buffer.output['time'] * self.interval)
        # 总共跑了多少时间 / 总共跑了多少 iter = 平均每个 iter 耗时
        time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1)
        # 剩余时间=平均每个 iter 时间× 剩余总 iter
        eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        runner.log_storage.append({'eta': eta_str})

    def append_max_memory(self, runner):
        mem = _get_max_memory(runner)
        if mem == 0:
            return
        runner.log_storage.append({'memory': mem})

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
        self._avg_buffer.update({'time': time.time() - self.t})
        self.t = time.time()

        if self.every_n_iters(runner, self.interval):
            # TODO: 自动找到所有 append 开头的函数自动执行
            self.append_lr(runner)
            self.append_runner_type(runner)
            self.append_time(runner)
            self.append_max_memory(runner)

            log_str = self._parse_log(runner.log_storage.values())
            Logger.info(log_str)
            # 每隔 self.interval 清空一次，也就是每次打印的 date_time 都是 self.interval 个值平均的
            self._avg_buffer.clear()

    def _parse_log(self, log_storage_value):
        log_items = []
        # TODO： support loss smooth print
        for values in log_storage_value:
            for name, val in values.items():
                if isinstance(val, float):
                    val = f'{val:.4f}'
                log_items.append(f'{name}: {val}')
        return ', '.join(log_items)
