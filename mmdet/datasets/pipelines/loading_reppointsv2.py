import numpy as np
import cv2
from ... import cv_core

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadRPDV2Annotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`cv_core.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self, num_classes=80):
        super(LoadRPDV2Annotations, self).__init__()
        self.num_classes = num_classes

    def _load_semantic_map_from_box(self, results):
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']
        pad_shape = results['pad_shape']
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_sem_map = np.zeros((self.num_classes, int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.float32)
        gt_sem_weights = np.zeros((self.num_classes, int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.float32)

        indexs = np.argsort(gt_areas)
        for ind in indexs[::-1]:
            box = gt_bboxes[ind]
            box_mask = np.zeros((int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.int64)
            box_mask[int(box[1] / 8):int(box[3] / 8) + 1, int(box[0] / 8):int(box[2] / 8) + 1] = 1
            gt_sem_map[gt_labels[ind]][box_mask > 0] = 1
            gt_sem_weights[gt_labels[ind]][box_mask > 0] = 1 / gt_areas[ind]

        results['gt_sem_map'] = gt_sem_map
        results['gt_sem_weights'] = gt_sem_weights

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        results = self._load_semantic_map_from_box(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox_semantic_map={True}, '
        return repr_str


@PIPELINES.register_module()
class LoadDenseRPDV2Annotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`cv_core.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self):
        super(LoadDenseRPDV2Annotations, self).__init__()

    def mask_to_poly(self, mask):
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                polygons.append(contour)
        return polygons

    def _load_semantic_map_from_mask(self, results):
        gt_bboxes = results['gt_bboxes']
        gt_masks = results['gt_masks'].masks
        gt_labels = results['gt_labels']
        pad_shape = results['pad_shape']
        gt_sem_map = np.zeros((int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.int64)

        for i in range(gt_bboxes.shape[0]):
            mask_rescale = cv_core.imrescale(gt_masks[i], 1. / 8, interpolation='nearest')
            gt_sem_map = np.maximum(gt_sem_map, mask_rescale * (gt_labels[i] + 1))
        gt_sem_map = gt_sem_map - 1

        gt_sem_map = gt_sem_map[None, ...]
        results['gt_sem_map'] = gt_sem_map

        return results

    def _load_contours(self, results):
        gt_bboxes = results['gt_bboxes']
        gt_masks = results['gt_masks'].masks
        gt_contour_map = np.zeros_like(gt_masks[0], dtype=np.uint8)

        for i in range(gt_bboxes.shape[0]):
            polygons = self.mask_to_poly(gt_masks[i])
            for poly in polygons:
                poly = np.array(poly).astype(np.int)
                for j in range(len(poly) // 2):
                    x_0, y_0 = poly[2 * j:2 * j + 2]
                    if j == len(poly) // 2 - 1:
                        x_1, y_1 = poly[0:2]
                    else:
                        x_1, y_1 = poly[2 * j + 2:2 * j + 4]
                    cv2.line(gt_contour_map, (x_0, y_0), (x_1, y_1), 1, thickness=2)

        gt_contours = np.stack([np.nonzero(gt_contour_map)[1], np.nonzero(gt_contour_map)[0]], axis=1).astype(np.float32)

        results['gt_contours'] = gt_contours

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        results = self._load_semantic_map_from_mask(results)
        results = self._load_contours(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_mask_semantic_map={True}, '
        repr_str += f'(with_contours={True}, '
        return repr_str
