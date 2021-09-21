

class ParamScheduler:
    def get_lr(self, engine,by_epoch):
        raise NotImplementedError("Param schedulers must override get_lr")


class LambdaParamScheduler(ParamScheduler):
    def __init__(self, scheduler_fun):
        super(LambdaParamScheduler,self).__init__()
        assert scheduler_fun is not None
        self.scheduler_fun=scheduler_fun


    def get_lr(self, engine,by_epoch):
        progress = engine.epoch if by_epoch else engine.iter
        return self.scheduler_fun(progress)



class ConstantParamScheduler(ParamScheduler):
    """
    Returns a constant value for a param.
    """

    def __init__(self, value):
        super(ConstantParamScheduler,self).__init__()
        self._value = value

    def get_lr(self, engine,by_epoch):
        return self._value



# 基于解耦原则，还是应该实现复合曲线类，不然自动根据个数在hook中组合，感觉会不好维护
class CompositeParamScheduler(ParamScheduler):
    pass







