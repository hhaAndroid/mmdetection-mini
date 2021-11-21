import math
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import copy


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 3, 3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.conv1(x)
        return x


class Mydataset(Dataset):
    def __init__(self):
        self.img = np.random.random((3, 100, 100))

    def __getitem__(self, item):
        return self.img

    def __len__(self):
        return 118287

    @staticmethod
    def collate_fn(batch):
        return np.array(batch)


def one_cycle(y1=0.0, y2=1.0, steps=100, epochs=300):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: (1 - x / (epochs - 1)) * (1.0 - 0.2) + 0.2
    # return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class Runner():
    def __init__(self, optimizer, iter, epoch, nb):
        self.optimizer = optimizer
        self.iter = iter
        self.epoch = epoch
        self.nb = nb


if __name__ == '__main__':
    model = Model()
    hyp = {'lr0': 0.01, 'lrf': 0.1, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
           'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5,
           'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0,
           'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0,
           'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
           'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0}

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    optimizer1 = copy.deepcopy(optimizer)
    # hook = OneCycleLrUpdaterHook()

    epochs = 300
    batch_size = 64
    total_batch_size = 64
    nbs = 64

    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    dataset = Mydataset()
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=Mydataset.collate_fn)
    nb = len(dataloader)
    # hook.before_run(Runner(optimizer1, 0, 1, nb))
    # 最多 warmup 1000 次迭代，实际上设置为 3 epoch
    nw = max(round(3 * nb), 1000)

    for epoch in range(0, epochs):
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=nb)
        for i, imgs in pbar:
            ni = i + nb * epoch
            # hook.before_train_iter(Runner(optimizer1, ni, epoch, nb))

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                print(epoch, ni,'yolov5=', ni, [x['lr'] for x in optimizer.param_groups])

        # 一个 epoch 才改变一次 lr
        # hook.after_train_epoch(Runner(optimizer1, 0, epoch,nb))
        scheduler.step()
        print(epoch, 'yolov5=', [x['lr'] for x in optimizer.param_groups])
