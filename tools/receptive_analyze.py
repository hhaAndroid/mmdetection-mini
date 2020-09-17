# -*- coding:utf-8 -*-
import os.path as osp
import sys
import argparse
from functools import partial

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from mmdet.cv_core import calc_receptive_filed, Config
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


def create_model(cfg):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    return model


def analyze_receptive(cfg, input_shape, index=1):
    model = create_model(cfg)
    model.forward = partial(forward, model)
    calc_receptive_filed(model, input_shape, index)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    input_shape = (608, 608, 3)
    index = 62
    analyze_receptive(cfg, input_shape, index)
