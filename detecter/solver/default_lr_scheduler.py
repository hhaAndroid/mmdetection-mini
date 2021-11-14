from torch.optim import Optimizer
from .builder import LR_SCHEDULERS
from cvcore import build_from_cfg
from .builder import PARAM_SCHEDULERS
import copy

__all__ = ['LRScheduler','build_default_lr_scheduler']



@LR_SCHEDULERS.register_module()
def build_default_lr_scheduler(optimizer,**lr_scheduler_cfg):
    return LRScheduler(optimizer, **lr_scheduler_cfg)



class LRScheduler:

    def __init__(self, optimizer, param_scheduler, param_steps, by_epoch=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))


        if not isinstance(param_scheduler,list):
            param_scheduler=[param_scheduler]

        assert len(param_scheduler)== len(param_steps) or len(param_scheduler)+1 == len(param_steps)
        if len(param_scheduler)== len(param_steps):
            param_steps.append(-1)

        cp_param_schedulers_cfg=copy.deepcopy(param_scheduler)

        param_schedulers=[]
        for i,param_scheduler_cfg in enumerate(cp_param_schedulers_cfg):
            if 'by_epoch' not in param_scheduler_cfg:
                param_scheduler_cfg['by_epoch']=by_epoch
            if 'begin' not in param_scheduler_cfg:
                param_scheduler_cfg['begin']=param_steps[i]
            if 'end' not in param_scheduler_cfg:
                param_scheduler_cfg['end']=param_steps[i+1]

            param_scheduler=build_from_cfg(param_scheduler_cfg,PARAM_SCHEDULERS)
            param_schedulers.append(param_scheduler)

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
            if scheduler.by_epoch:
                if scheduler.end == -1: scheduler.end=runner.max_epochs

                if scheduler.begin <= runner.epoch <= scheduler.end:
                    values = [scheduler.step(runner, base_lr, index) for index,base_lr in enumerate(self.base_lrs)]
                    break

            else:
                if scheduler.end == -1: scheduler.end = runner.max_iters

                if scheduler.begin <= runner.iter <= scheduler.end:
                    values = [scheduler.step(runner, base_lr, index) for index,base_lr in enumerate(self.base_lrs)]
                    break

        assert values is not None
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data

            if isinstance(lr,dict):
                param_group.update(lr)
            else:
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
