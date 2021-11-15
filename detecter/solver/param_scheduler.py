import cvcore
from .builder import PARAM_SCHEDULERS

__all__ = ['ParamScheduler', 'LambdaParamScheduler', 'ConstantParamScheduler',
           'StepParamScheduler', 'LinearParamScheduler',
           'Yolov5WramUpParamScheduler', 'Yolov5OneCycleParamScheduler']


class ParamScheduler:
    def __init__(self, by_epoch, begin, end):
        self.by_epoch = by_epoch
        self.begin = begin
        self.end = end

    def step(self, runner, base_lr, index):
        raise NotImplementedError("Param schedulers must override get_lr")


@PARAM_SCHEDULERS.register_module()
class LambdaParamScheduler(ParamScheduler):
    def __init__(self, scheduler_fun, **kwargs):
        super(LambdaParamScheduler, self).__init__(**kwargs)
        assert scheduler_fun is not None
        self.scheduler_fun = scheduler_fun

    def step(self, runner, base_lr, index):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.scheduler_fun(progress)


@PARAM_SCHEDULERS.register_module()
class ConstantParamScheduler(ParamScheduler):
    """
    Returns a constant value for a param.
    """

    def __init__(self, value, **kwargs):
        super(ConstantParamScheduler, self).__init__(**kwargs)
        self._value = value

    def step(self, runner, base_lr, index):
        return base_lr * self._value


@PARAM_SCHEDULERS.register_module()
class StepParamScheduler(ParamScheduler):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, step, gamma=0.1, min_lr=None, **kwargs):
        super(StepParamScheduler, self).__init__(**kwargs)
        if isinstance(step, list):
            assert cvcore.is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self._step = step
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, runner, base_lr, index):
        # progress 这个参数应该是在外面计算好传入，而不是在每个类调用时候判断
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self._step, int):
            exp = progress // self._step
        else:
            exp = len(self._step)
            for i, s in enumerate(self._step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma ** exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr


@PARAM_SCHEDULERS.register_module()
class LinearParamScheduler(ParamScheduler):
    def __init__(self, start_value=0, end_value=1, **kwargs):
        super(LinearParamScheduler, self).__init__(**kwargs)
        self._start_value = start_value
        self._end_value = end_value

    def step(self, runner, base_lr, index):
        progress = runner.epoch if self.by_epoch else runner.iter
        assert self.begin <= progress <= self.end
        where = (progress - self.begin) / (self.end - self.begin)
        # interpolate between start and end values
        return base_lr * (self._end_value * where + self._start_value * (1 - where))

import math
import numpy as np

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


@PARAM_SCHEDULERS.register_module()
class Yolov5WramUpParamScheduler(ParamScheduler):
    def __init__(self, by_epoch, begin, end):
        super().__init__(by_epoch, begin, end)
        self.total_iters=end
        self.momentum = 0.937
        self.warmup_bias_lr = 0.1
        self.warmup_momentum = 0.8


    def step(self, runner, base_lr, index):
        xi = [0, self.total_iters]
        one_cycle_fun=one_cycle(1, 0.2, runner.max_epochs)

        cur_epoch=runner.epoch
        cur_iters=runner.iter

        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        x= {'lr': np.interp(cur_iters, xi,
                            [self.warmup_bias_lr if index == 2 else 0.0, base_lr * one_cycle_fun(cur_epoch)]),
            'momentum': np.interp(cur_iters, xi, [self.warmup_momentum, self.momentum])}

        return x



@PARAM_SCHEDULERS.register_module()
class Yolov5OneCycleParamScheduler(ParamScheduler):
    def step(self, runner, base_lr,index):
        one_cycle_fun = one_cycle(1, 0.2, runner.max_epochs)
        return base_lr * one_cycle_fun(runner.epoch)

