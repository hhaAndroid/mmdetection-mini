import torch

from ..builder import BBOX_ASSIGNERS
from .base_assigner import BaseAssigner
from .assign_result import AssignResult


@BBOX_ASSIGNERS.register_module()
class PointHMAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, gaussian_bump=False, gaussian_iou=0.7):
        self.gaussian_bump = gaussian_bump
        self.gaussian_iou = gaussian_iou

    def assign(self, points, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every point, each bbox
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. for each gt box, we find the k most closest points to the
            box center and assign the gt bbox to those points, we also record
            the minimum distance from each point to the closest gt box. When we
            assign the bbox to the points, we check whether its distance to the
            points is closest.

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 1e8
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        if self.gaussian_bump:
            dtype = torch.float32
        else:
            dtype = torch.long
        if num_points == 0 or num_gts == 0:
            assigned_gt_hm_tl = points.new_zeros((num_points,), dtype=dtype)
            assigned_gt_hm_br = points.new_zeros((num_points,), dtype=dtype)
            # stores the assigned gt dist (to this point) of each point
            assigned_gt_offset_tl = points.new_zeros((num_points, 2), dtype=torch.float32)
            assigned_gt_offset_br = points.new_zeros((num_points, 2), dtype=torch.float32)

            pos_inds_tl = torch.nonzero(assigned_gt_hm_tl == 1, as_tuple=False).squeeze(-1).unique()
            pos_inds_br = torch.nonzero(assigned_gt_hm_br == 1, as_tuple=False).squeeze(-1).unique()
            neg_inds_tl = torch.nonzero(assigned_gt_hm_tl < 1, as_tuple=False).squeeze(-1).unique()
            neg_inds_br = torch.nonzero(assigned_gt_hm_br < 1, as_tuple=False).squeeze(-1).unique()

            return assigned_gt_hm_tl, assigned_gt_offset_tl, pos_inds_tl, neg_inds_tl, \
                   assigned_gt_hm_br, assigned_gt_offset_br, pos_inds_br, neg_inds_br
        points_range = torch.arange(num_points)
        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt box
        gt_bboxes_xtl, gt_bboxes_ytl, gt_bboxes_xbr, gt_bboxes_ybr = torch.chunk(gt_bboxes, 4, dim=1)
        gt_bboxes_xytl = torch.cat([gt_bboxes_xtl, gt_bboxes_ytl], -1)
        gt_bboxes_xybr = torch.cat([gt_bboxes_xbr, gt_bboxes_ybr], -1)
        if self.gaussian_bump:
            gt_bboxes_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            gt_bboxes_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            radius = gaussian_radius((gt_bboxes_h, gt_bboxes_w), self.gaussian_iou)
            diameter = 2 * radius + 1
            sigma = diameter / 6
        else:
            radius = None

        distances_tl = (points_xy[:, None, :] - gt_bboxes_xytl[None, :, :]).norm(dim=2)
        distances_br = (points_xy[:, None, :] - gt_bboxes_xybr[None, :, :]).norm(dim=2)

        # stores the assigned gt heatmap of each point
        assigned_gt_hm_tl = points.new_zeros((num_points,), dtype=dtype)
        assigned_gt_hm_br = points.new_zeros((num_points,), dtype=dtype)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_offset_tl = points.new_zeros((num_points, 2), dtype=torch.float32)
        assigned_gt_offset_br = points.new_zeros((num_points, 2), dtype=torch.float32)

        lvls = torch.arange(lvl_min, lvl_max + 1, dtype=points_lvl.dtype, device=points_lvl.device)
        for gt_lvl in lvls:
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            lvl_points = points_xy[lvl_idx, :]

            downscale_factor = torch.pow(2, gt_lvl)
            lvl_distances_tl = distances_tl[lvl_idx, :]
            lvl_distances_br = distances_br[lvl_idx, :]

            _, min_dist_index_tl = lvl_distances_tl.min(dim=0)
            min_dist_points_index_tl = points_index[min_dist_index_tl]

            assigned_gt_offset_tl[min_dist_points_index_tl, :] = \
                (gt_bboxes_xytl - lvl_points[min_dist_index_tl, :]) / downscale_factor

            _, min_dist_index_br = lvl_distances_br.min(dim=0)
            min_dist_points_index_br = points_index[min_dist_index_br]
            assigned_gt_offset_br[min_dist_points_index_br, :] = \
                (gt_bboxes_xybr - lvl_points[min_dist_index_br, :]) / downscale_factor
            if self.gaussian_bump:
                out_index_tl = lvl_distances_tl >= radius[None, :]
                lvl_gaussian_tl = torch.exp(-torch.pow(lvl_distances_tl, 2) / (2 * sigma * sigma)[None, :])
                lvl_gaussian_tl[out_index_tl] = -INF
                max_gaussian_tl, _ = lvl_gaussian_tl.max(dim=1)
                assigned_gt_hm_tl[points_index[max_gaussian_tl != -INF]] = max_gaussian_tl[max_gaussian_tl != -INF]

                out_index_br = lvl_distances_br >= radius[None, :]
                lvl_gaussian_br = torch.exp(-torch.pow(lvl_distances_br, 2) / (2 * sigma * sigma)[None, :])
                lvl_gaussian_br[out_index_br] = -INF
                max_gaussian_br, _ = lvl_gaussian_br.max(dim=1)
                assigned_gt_hm_br[points_index[max_gaussian_br != -INF]] = max_gaussian_br[max_gaussian_br != -INF]
            assigned_gt_hm_tl[min_dist_points_index_tl] = 1
            assigned_gt_hm_br[min_dist_points_index_br] = 1

        pos_inds_tl = torch.nonzero(assigned_gt_hm_tl == 1, as_tuple=False).squeeze(-1).unique()
        pos_inds_br = torch.nonzero(assigned_gt_hm_br == 1, as_tuple=False).squeeze(-1).unique()
        neg_inds_tl = torch.nonzero(assigned_gt_hm_tl < 1, as_tuple=False).squeeze(-1).unique()
        neg_inds_br = torch.nonzero(assigned_gt_hm_br < 1, as_tuple=False).squeeze(-1).unique()

        return assigned_gt_hm_tl, assigned_gt_offset_tl, pos_inds_tl, neg_inds_tl, \
               assigned_gt_hm_br, assigned_gt_offset_br, pos_inds_br, neg_inds_br


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)

    r = torch.stack([r1, r2, r3], dim=1)
    return torch.min(r, dim=1)[0]