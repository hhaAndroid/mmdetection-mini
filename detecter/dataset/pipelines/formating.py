from ..builder import PIPELINES
from ...core.structures import Instances
import warnings

__all__ = ['Collect']


@PIPELINES.register_module()
class Collect:
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

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
