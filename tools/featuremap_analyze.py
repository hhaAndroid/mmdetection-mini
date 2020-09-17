import argparse
import os
from pathlib import Path
from functools import partial
import cv2
import numpy as np
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from mmdet.cv_core import (Config, load_checkpoint, FeatureMapVis, show_tensor, imdenormalize, show_img, imwrite,
                           traverse_file_paths)
from mmdet.models import build_detector
from mmdet.datasets.builder import build_dataset
from mmdet.datasets.pipelines import Compose


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_dir', type=str, default='../demo', help='show img dir')
    # 显示预测结果
    parser.add_argument('--show', type=bool, default=True, help='show results')
    # 可视化图片保存路径
    parser.add_argument(
        '--output_dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def forward(self, img, img_metas=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    return outs


def create_model(cfg, use_gpu=True):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if use_gpu:
        model = model.cuda()
    return model


def create_featuremap_vis(cfg, use_gpu=True, init_shape=(320, 320, 3)):
    model = create_model(cfg, use_gpu)
    model.forward = partial(forward, model)
    featurevis = FeatureMapVis(model, use_gpu)
    featurevis.set_hook_style(init_shape[2], init_shape[:2])
    return featurevis


def _show_save_data(featurevis, img, img_orig, feature_indexs, filepath, is_show, output_dir):
    show_datas = []
    for feature_index in feature_indexs:
        feature_map = featurevis.run(img.copy(), feature_index=feature_index)[0]
        data = show_tensor(feature_map[0], resize_hw=img.shape[:2], show_split=False, is_show=False)[0]
        am_data = cv2.addWeighted(data, 0.5, img_orig, 0.5, 0)
        show_datas.append(am_data)
    if is_show:
        show_img(show_datas)
    if output_dir is not None:
        filename = os.path.join(output_dir,
                                Path(filepath).name
                                )
        if len(show_datas) == 1:
            imwrite(show_datas[0], filename)
        else:
            for i in range(len(show_datas)):
                fname, suffix = os.path.splitext(filename)
                imwrite(show_datas[i], fname + '_{}'.format(str(i)) + suffix)


def show_featuremap_from_imgs(featurevis, feature_indexs, img_dir, mean, std, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    img_paths = traverse_file_paths(img_dir, 'jpg')
    for path in img_paths:
        data = dict(img_info=dict(filename=path), img_prefix=None)
        test_pipeline = Compose(cfg.data.test.pipeline)
        item = test_pipeline(data)
        img_tensor = item['img']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # 依然是归一化后的图片
        img_orig = imdenormalize(img, np.array(mean), np.array(std)).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, path, is_show, output_dir)


def show_featuremap_from_datalayer(featurevis, feature_indexs, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    dataset = build_dataset(cfg.data.test)
    for item in dataset:
        img_tensor = item['img']
        img_metas = item['img_metas'][0].data
        filename = img_metas['filename']
        img_norm_cfg = img_metas['img_norm_cfg']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # 依然是归一化后的图片
        img_orig = imdenormalize(img, img_norm_cfg['mean'], img_norm_cfg['std']).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, filename, is_show, output_dir)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    use_gpu = False
    is_show = args.show
    init_shape = (320, 320, 3)  # 值不重要，只要前向一遍网络时候不报错即可
    feature_index = [218, 214, 210]  # 想看的特征图层索引(yolov3  218 214 210)

    featurevis = create_featuremap_vis(cfg, use_gpu, init_shape)
    # show_featuremap_from_datalayer(featurevis, feature_index, is_show, args.output_dir)

    mean = cfg.img_norm_cfg['mean']
    std = cfg.img_norm_cfg['std']
    show_featuremap_from_imgs(featurevis, feature_index, args.img_dir, mean, std, is_show, args.output_dir)
