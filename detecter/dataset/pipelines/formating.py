from ..builder import PIPELINES
from ...core.structures import Instances, Boxes
import warnings
import torch
import numpy as np

__all__ = ['Collect', 'ToTensor']


@PIPELINES.register_module()
class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        results["img"] = torch.as_tensor(np.ascontiguousarray(results['img'].transpose(2, 0, 1)))
        for key in self.keys:
            results[key] = torch.as_tensor(results[key])
        return results


@PIPELINES.register_module()
class Collect:
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results.
        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """
        data = {'img': results['img']}

        instance = Instances(results['img_shape'])
        for key in self.keys:
            instance._fields[key] = results[key]

        if 'gt_bboxes' in self.keys:
            instance.gt_bboxes = Boxes(results['gt_bboxes'])

        data['annotations'] = instance

        img_meta = {}
        for key in self.meta_keys:
            if key not in results:
                warnings.warn(f'{key} not in results. It has skip.')
            img_meta[key] = results[key]
        data['img_meta'] = img_meta

        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
