# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
import copy
import itertools

import numpy as np
from torch.utils.data import Dataset
from tabulate import tabulate

from cvcore import Logger
from .builder import DATASETS
from .pipelines import Compose

__all__ = ['CustomDataset']


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        classes = entry['ann_info']['labels']
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    Logger.info("Distribution of instances among all {} categories:\n".format(num_classes))
    Logger.info('\n'+table)


@DATASETS.register_module()
class CustomDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 train_mode=True,
                 should_load_anas=True,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.train_mode = train_mode
        self.should_load_anas = should_load_anas
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

        # 为了后续方便，不管啥数据集，都必须要有 image_id 唯一 key
        assert all(['image_id' in info['img_info'] for info in self.data_infos])

        # filter images too small and containing no annotations
        if self.train_mode:
            num_before = len(self.data_infos)
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            num_after = len(self.data_infos)
            Logger.warn(
                "Removed {} images with no usable annotations. {} images left.".format(
                    num_before - num_after, num_after
                )
            )

        assert len(self.data_infos) > 0, 'dataset is empty'
        # processing pipeline
        self.pipeline = Compose(pipeline)

        has_ann = "ann_info" in self.data_infos[0]
        if has_ann:
            print_instances_class_histogram(self.data_infos, self.CLASSES)

        mode = "training" if self.train_mode else "inference"
        Logger.debug(f"Pipelines used in {mode}: {self.pipeline}")

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        pass

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, result in enumerate(self.data_infos):
            img_info = result['img_info']
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        return np.random.choice(len(self))

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if not self.train_mode:
            return self.prepare_img(idx)
        while True:
            data = self.prepare_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        results = copy.deepcopy(self.data_infos[idx])
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_data_infos(self):
        return self.data_infos

    # 重要
    def get_global_metas(self):
        return {"class": self.CLASSES}
