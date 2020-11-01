import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class YOLOBBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6, scale_x_y=1.0):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps
        self.scale_x_y = scale_x_y

    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        # 所有值都是原图坐标值，故需要/ stride，变成特征图尺度
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # w h编码是 gt bbox的宽高/anchor宽高，加上log，
        # 保证大小bbox的宽高预测值不会差距很大(log可以拉近差距)，保证训练过程更加稳定而已
        w_target = torch.log((w_gt / w).clamp(min=self.eps))
        h_target = torch.log((h_gt / h).clamp(min=self.eps))
        # xy编码就是gt中心基于左上角网格点的偏移
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        encoded_bboxes = torch.stack(
            [x_center_target, y_center_target, w_target, h_target], dim=-1)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # Get outputs x, y
        x_center_pred = (pred_bboxes[..., 0] * self.scale_x_y - 0.5 * (self.scale_x_y - 1) - 0.5) * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] * self.scale_x_y - 0.5 * (self.scale_x_y - 1) - 0.5) * stride + y_center
        w_pred = torch.exp(pred_bboxes[..., 2]) * w
        h_pred = torch.exp(pred_bboxes[..., 3]) * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)

        return decoded_bboxes
