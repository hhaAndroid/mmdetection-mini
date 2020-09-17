# -*- coding:utf-8 -*-
import os.path as osp
import sys
import torch
import argparse
from functools import partial

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from mmdet.cv_core import Timer, Config
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def forward(self, img, img_metas=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    return outs


def create_model(cfg, use_gpu=True):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    if use_gpu:
        model = model.cuda()
    return model


def analyze_time_random(cfg, use_gpu, count=1000, input_shape=(2, 3, 256, 256)):
    model = create_model(cfg, use_gpu)
    model.forward = partial(forward, model)
    input_tensor = torch.rand(input_shape)
    if use_gpu:
        input_tensor = input_tensor.cuda()
    with torch.no_grad():
        _ = model(input_tensor)  # 第一次不算
        infer_time = Timer()
        for _ in range(int(count)):
            _ = model(input_tensor)
        average_time = infer_time.since_start()/count
    batch_avg_time = round(average_time * 1000, 5)
    one_avg_time = round(average_time * 1000 / input_shape[0], 5)
    print('batch_avg_time={}ms,one_avg_time=={}ms'.format(batch_avg_time, one_avg_time))


def analyze_time_datalayer(cfg, use_gpu, count=1000, is_load_weights=False):
    pass


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    use_gpu = True
    input_shape = (2, 3, 320, 320)
    analyze_time_random(cfg, use_gpu=use_gpu, input_shape=input_shape)
