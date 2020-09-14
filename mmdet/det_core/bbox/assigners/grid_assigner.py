import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

# 整个算法流程和max_iou_assigner非常类似，只不过多了一个box_responsible_flags限制
# 所以可以看出，其分配测试和原版yolov3有差别，主要是本文策略会引入更多正样本，可能能加速收敛
@BBOX_ASSIGNERS.register_module()
class GridAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, box_responsible_flags, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes. The process is very much like the max iou
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts <= neg_iou_thr to 0
        3. for each bbox within a cell, if the iou with its nearest gt >
            pos_iou_thr and the center of that gt falls inside the cell,
            assign it to that bbox
        4. for each gt bbox, assign its nearest proposals within the cell the
            gt bbox falls in to itself.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            box_responsible_flags (Tensor): flag to indicate whether box is
                responsible for prediction, shape(n, )
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all gt and bboxes
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        # 1. assign -1 by default 先把所有anchor位置设置为忽略
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_bboxes
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)  # 对于每个anchor，找到和他最匹配的gt iou

        # 将所有anchor位置的max_overlaps阈值小于阈值的，全部认为是负样本
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps <= self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps > self.neg_iou_thr[0])
                             & (max_overlaps <= self.neg_iou_thr[1])] = 0

        # 3. assign positive: falls into responsible cell and above
        # positive IOU threshold, the order matters.
        # the prior condition of comparision is to filter out all
        # unrelated anchors, i.e. not box_responsible_flags
        # 将没有gt bbox中心位置的anchor iou强制设置为最小值-1
        # 意思是为了避免：gt bbox中心落在A格子内部，但是最大Iou对应的anchor不在A,而是B，不允许出现这种情况
        # 虽然出现这种情况也是可以训练的，但是这不符合yolo思想，因为我们是强制落在网格处的才负责预测
        overlaps[:, ~box_responsible_flags.type(torch.bool)] = -1.

        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)  # 对于每个anchor，找到和他最匹配的gt iou

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)  # 对于每个gt，找到和他最匹配的anchor

        # 对于所有anchor中的每一个位置，如果该位置和gt bbox的iou大于阈值，并且gt bbox的中心落在对应的anchor处
        # 这个操作会导致：仅仅处于gt bbox落在的网格中心的anchor，并且iou要大，才会匹配
        # 和yolo原论文有点区别：原始论文是每个gt bbox一定有一个唯一的anchor进行对应，不存在多个anchor对应1个Gt
        # 而这里，可能存在，因为任何一个网络中心都有三个anchor，如果这三个都大于self.pos_iou_thr，那就有可能
        # 这里，可能有些gt bbox还没有anchor匹配，后面会解决
        pos_inds = (max_overlaps >
                    self.pos_iou_thr) & box_responsible_flags.type(torch.bool)
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1  # 正样本anchor，保存对应的gt bbox索引，表示和其匹配上

        # 4. assign positive to max overlapped anchors within responsible cell
        # 遍历每个Gt,判断和他iou最大的anchor，是否大于self.min_pos_iou=0阈值
        # 这里可以保证每个gt一定有anchor进行匹配
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    # 找出anchor索引，一定要满足box_responsible_flags才行
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & \
                         box_responsible_flags.type(torch.bool)
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif box_responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # assign labels of positive anchors
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]  # 正样本anchor处的label值

        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
