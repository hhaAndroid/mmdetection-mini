import time
import os.path as osp
import torch

import cvcore
from cvcore import build_from_cfg, HOOKS, Logger
from cvcore.utils import dist_comm

from ..solver import build_optimizer, build_lr_scheduler
from ..model import build_detector
from ..dataset import build_dataset
from ..dataloader import build_dataloader
from .builder import build_runner
from ..utils import collect_env, set_random_seed, auto_replace_data_root, wrapper_model

__all__ = ['DefaultTrainer']


class DefaultTrainer:
    def __init__(self, cfg):
        self.logger = None
        self.meta = None
        # The order is more critical
        self.cfg = auto_replace_data_root(cfg)
        self.setup_cfg()
        self.setup_logger()
        self.setup_seed()
        self.setup_envs()

        self.detector = wrapper_model(self.build_detector())
        self.train_dataset = self.build_train_dataset()
        self.train_dataloader = self.build_train_dataloader()
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_lr_scheduler()

        self.val_dataset = None
        self.val_dataloader = None

    def setup_cfg(self):
        # import modules from string list.
        if self.cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**self.cfg['custom_imports'])
            # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if self.cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            self.cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(self.cfg.config))[0])
        cvcore.mkdir_or_exist(self.cfg.work_dir)

    def setup_logger(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.cfg.work_dir, f'{timestamp}.log')
        logger_cfg = self.cfg.logger
        if 'log_file' not in logger_cfg:
            logger_cfg['log_file'] = log_file
        self.logger = Logger.init(logger_cfg)

    def setup_seed(self):
        if self.cfg.get('seed', None) is not None:
            self.logger.info(f'Set random seed to {self.cfg.seed}, '
                             f'deterministic: {self.cfg.deterministic}')
            set_random_seed(self.cfg.seed, deterministic=self.cfg.deterministic)

    def setup_envs(self):
        # create work_dir
        cvcore.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        # dump config
        self.cfg.dump(osp.join(self.cfg.work_dir, osp.basename(self.cfg.config)))

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                         dash_line)
        meta['env_info'] = env_info
        meta['config'] = self.cfg.pretty_text
        # log some basic info
        self.logger.info(f'Config:\n{self.cfg.pretty_text}')

        is_distributed = dist_comm.get_world_size() > 1
        self.logger.info(f'Distributed training: {is_distributed}, '
                         f'world size: {dist_comm.get_world_size()}')

        if self.cfg.get('seed', None) is not None:
            meta['seed'] = self.cfg.seed
        meta['exp_name'] = osp.basename(self.cfg.config)
        self.meta = meta

    def build_detector(self, detector=None):
        if detector is not None:
            return detector

        if 'vis_interval' in self.cfg:
            detector = build_detector(self.cfg.model, dict(vis_interval=self.cfg.vis_interval))
        else:
            detector = build_detector(self.cfg.model)
        detector.init_weights()
        return detector

    def build_train_dataset(self, dataset=None):
        if dataset is not None:
            return dataset
        return build_dataset(self.cfg.data.train)

    def build_train_dataloader(self, dataloader=None):
        if dataloader is not None:
            return dataloader
        return build_dataloader(self.cfg.dataloader.train, self.train_dataset)

    def build_val_dataset(self, dataset=None):
        if dataset is not None:
            return dataset
        return build_dataset(self.cfg.data.val)

    def build_val_dataloader(self, dataloader=None):
        if dataloader is not None:
            return dataloader
        return build_dataloader(self.cfg.dataloader.val, self.val_dataset)

    def build_optimizer(self, optimizer=None):
        if optimizer is not None:
            return optimizer
        return build_optimizer(self.cfg.optimizer, self.detector)

    def build_lr_scheduler(self, lr_scheduler=None):
        if lr_scheduler is not None:
            return lr_scheduler
        return build_lr_scheduler(self.cfg.lr_scheduler, self.optimizer)

    def run(self):
        dataloaders = [self.train_dataloader]
        if len(self.cfg.workflow) == 2:
            self.val_dataset = self.build_val_dataset()
            self.val_dataloader = self.build_val_dataloader()
            dataloaders.append(self.val_dataloader)

        default_args = dict(model=self.detector,
                            optimizer=self.optimizer,
                            scheduler=self.lr_scheduler,
                            meta=self.meta,
                            logger=self.logger,
                            cfg=self.cfg,
                            work_dir=self.cfg.work_dir)
        runner = build_runner(self.cfg.runner, default_args)

        # user-defined hooks
        if self.cfg.get('custom_hooks', None):
            custom_hooks = self.cfg.custom_hooks
            assert isinstance(custom_hooks, list), \
                f'custom_hooks expect list type, but got {type(custom_hooks)}'
            for hook_cfg in self.cfg.custom_hooks:
                assert isinstance(hook_cfg, dict), \
                    'Each item in custom_hooks expects dict type, but got ' \
                    f'{type(hook_cfg)}'
                hook_cfg = hook_cfg.copy()
                priority = hook_cfg.pop('priority', 'NORMAL')
                hook = build_from_cfg(hook_cfg, HOOKS)
                runner.register_hook(hook, priority=priority)

        if 'resume_from' in self.cfg and self.cfg.resume_from:
            runner.resume_or_load(self.cfg.resume_from, resume=True)
        elif 'load_from' in self.cfg and self.cfg.load_from:
            runner.resume_or_load(self.cfg.load_from, resume=False)

        runner.run(dataloaders, self.cfg.workflow)
