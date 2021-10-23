from torch.optim import Optimizer
from .builder import LR_SCHEDULERS
from cvcore import build_from_cfg
from .builder import PARAM_SCHEDULERS

__all__ = ['LRScheduler','build_default_lr_scheduler']



@LR_SCHEDULERS.register_module()
def build_default_lr_scheduler(optimizer,param_schedulers_cfg):
    if not isinstance(param_schedulers_cfg, list):
        param_schedulers_cfg = [param_schedulers_cfg]

    param_schedulers=[]
    for param_scheduler_cfg in param_schedulers_cfg:
        param_schedulers.append(build_from_cfg(param_scheduler_cfg,PARAM_SCHEDULERS))

    return LRScheduler(optimizer,param_schedulers)



class LRScheduler(object):

    def __init__(self, optimizer, param_schedulers):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

        if not isinstance(param_schedulers,list):
            param_schedulers=[param_schedulers]
        self.param_schedulers=param_schedulers

        self.last_epoch = -1


    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        save_dict={}
        for i, param_scheduler in enumerate(self.param_schedulers):
            save_dict[i]=param_scheduler.state_dict()
        return save_dict


    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        for pstate_dict,param_scheduler in zip(state_dict,self.param_schedulers):
            param_scheduler.load_state_dict(pstate_dict)


    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr


    def step(self, runner):
        values=None
        for scheduler in self.param_schedulers:
            if scheduler['by_epoch']:
                if scheduler.begin <= runner.epoch <= scheduler.end:
                    values = [scheduler.step(runner, base_lr) for base_lr in self.base_lrs]
                    break

            else:
                if scheduler.begin <= runner.iter <= scheduler.end:
                    values = [scheduler.step(runner, base_lr) for base_lr in self.base_lrs]
                    break

        assert values is not None
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
