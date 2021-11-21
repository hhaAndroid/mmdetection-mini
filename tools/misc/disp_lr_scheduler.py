import argparse
from cvcore import Config, HOOKS, Hook, Logger
from cvcore import DictAction, get_best_param_group_id
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from detecter import build_dataloader, build_optimizer, build_lr_scheduler, build_runner
from loguru import logger
import copy
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(torch.stack(x, dim=0).permute(0, 3, 1, 2))
        x = self.bn(x)
        x = self.relu(x).mean()  # scaler
        return x


class DemoDataset(Dataset):
    def __getitem__(self, item):
        return torch.ones((10, 10, 1))

    def __len__(self):
        return 5000


class DispLRHook(Hook):
    def __init__(self):
        self.history = []

    def after_train_iter(self, runner):
        # _best_param_group_id = get_best_param_group_id(runner.optimizer)
        # lr = runner.optimizer.param_groups[_best_param_group_id]["lr"]
        lr=[x['lr'] for x in runner.optimizer.param_groups]

        Logger.info(f'epoch={runner.epoch},iter={runner.iter},lr={lr}')
        # self.history.append(lr)
        # Logger.info(f'{runner.iter}={lr}')

    def after_run(self, runner):
        pass
        # plt.plot(self.history)
        # plt.show()


@logger.catch
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.checkpoint.period=100000


    detector = DemoModel()
    optimizer = build_optimizer(cfg.optimizer, detector)
    scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer)

    cfg.pop('evaluator')
    cp_train_cfg = copy.deepcopy(cfg.dataloader.train)
    cp_train_cfg['aspect_ratio_grouping'] = False
    train_dataloader = build_dataloader(cp_train_cfg, DemoDataset())

    default_args = dict(model=detector, optimizer=optimizer,
                        scheduler=scheduler, logger=Logger.init(),cfg=cfg)
    runner = build_runner(cfg.runner, default_args)
    runner.register_hook(DispLRHook(), 100)
    runner.run([train_dataloader], [('train',1)])


if __name__ == '__main__':
    main()
