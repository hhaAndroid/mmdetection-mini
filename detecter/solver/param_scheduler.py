import cvcore
from .builder import PARAM_SCHEDULERS

__all__ = ['ParamScheduler', 'LambdaParamScheduler', 'ConstantParamScheduler',
           'StepParamScheduler', 'LinearParamScheduler']


class ParamScheduler:
    def __init__(self, by_epoch):
        self.by_epoch = by_epoch

    def __call__(self, runner, base_lr):
        raise NotImplementedError("Param schedulers must override get_lr")


@PARAM_SCHEDULERS.register_module()
class LambdaParamScheduler(ParamScheduler):
    def __init__(self, scheduler_fun, **kwargs):
        super(LambdaParamScheduler, self).__init__(**kwargs)
        assert scheduler_fun is not None
        self.scheduler_fun = scheduler_fun

    def __call__(self, runner, base_lr):
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

    def __call__(self, runner, base_lr):
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
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr

    def __call__(self, runner, base_lr):
        # progress 这个参数应该是在外面计算好传入，而不是在每个类调用时候判断
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
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
    def __init__(self, start_step, end_step, start_value=0, end_value=1, **kwargs):
        super(LinearParamScheduler, self).__init__(**kwargs)
        assert end_step > start_step
        self._start_step = start_step
        self._end_step = end_step
        self._start_value = start_value
        self._end_value = end_value

    def __call__(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        assert self._start_step <= progress <= self._end_step
        where = (progress - self._start_step) / (self._end_step - self._start_step)
        # interpolate between start and end values
        return base_lr * (self._end_value * where + self._start_value * (1 - where))

