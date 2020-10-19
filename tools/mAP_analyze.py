# -*- coding:utf-8 -*-
# deep eyes
import numpy as np
import torch
from mmdet.det_core.utils import voc_eval_map


def demo_voc2012_map():
    # gt_bbox1 = torch.tensor([[10.0, 20, 40, 60],
    #                          [50.0, 70, 90, 95]]).reshape(-1, 4)  # xyxy
    # pred_bbox1 = torch.tensor([[11.0, 17, 44, 85, 0.9],
    #                            [10, 27, 60, 80, 0.8],
    #                            [5, 20, 45, 85, 0.45],
    #
    #                            [50, 70, 80, 95, 0.8],
    #                            [40, 60, 60, 91, 0.6],
    #                            [45, 67, 88, 90, 0.3],
    #                            ])
    gt_bbox1 = torch.tensor([10.0, 20, 40, 60]).reshape(-1, 4)  # xyxy
    pred_bbox1 = torch.tensor([[11.0, 17, 48, 65, 0.9],
                               [10, 27, 60, 70, 0.8],
                               [5, 20, 45, 85, 0.45],
                               ])
    results = [[pred_bbox1]]
    annotations = [[gt_bbox1]]
    voc2012_map = voc_eval_map(results, annotations, name='voc2012', iou_thr=0.5)
    print('voc2012_map=', voc2012_map)


if __name__ == '__main__':
    demo_voc2012_map()
