import numpy as np
import copy

from .builder import DATASETS
from .coco import CocoDataset

__all__ = ['YOLOV5CocoDataset']


@DATASETS.register_module()
class YOLOV5CocoDataset(CocoDataset):
    def __init__(self,
                 *args, with_rectangular=True, img_size=640, batch_size=1, stride=32, pad=0.0, **kwargs):
        super(CocoDataset, self).__init__(*args, **kwargs)
        self.with_rectangular = with_rectangular

        if self.with_rectangular:

            image_shapes = self._calc_batch_shape()
            image_shapes = np.array(image_shapes, dtype=np.float64)

            n = len(image_shapes)  # number of images
            bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
            nb = bi[-1] + 1  # number of batches
            self.batch = bi  # batch index of image

            ar = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
            irect = ar.argsort()

            self.data_infos = [self.data_infos[i] for i in irect]

            ar = ar[irect]
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride


    def _calc_batch_shape(self):
        batch_shape = []
        for data_info in self.data_infos:
            img_info = data_info['img_info']
            batch_shape.append((img_info['width'], img_info['height']))
        return batch_shape

    def prepare_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        results = copy.deepcopy(self.data_infos[idx])
        results['img_info']['batch_shape'] = self.batch_shapes[self.batch[idx]]
        self.pre_pipeline(results)
        return self.pipeline(results)
