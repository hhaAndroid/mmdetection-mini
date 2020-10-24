from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmdet.cv_core.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # 核心操作，在训练过程中需要返回预测结果的，此时需要proposal_cfg不为none
        # 例如RPN head，其需要给rcnn进行refine
        if proposal_cfg is None:
            return losses
        else:
            # get_bboxes其实就是前向推理流程
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
