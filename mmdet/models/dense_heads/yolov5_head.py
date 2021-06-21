# -*- coding:utf-8 -*-
import math
import functools
import torch
from mmcv.ops import batched_nms
import torch.nn as nn
from ..utils import brick as vn_layer
from ..builder import HEADS
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from .yolo_head import YOLOV3Head
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(_make_divisible, divisor=divisor, width_multiple=width_multiple)


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


@HEADS.register_module()
class YOLOV5Head(YOLOV3Head):
    # 为了不传入新的参数，默认将self.out_channels=[depth_multiple,width_multiple]

    def _init_layers(self):
        model = []

        make_div8_fun = make_divisible(8, self.out_channels[1])
        make_round_fun = make_round(self.out_channels[0])

        conv1 = vn_layer.Conv(make_div8_fun(1024), make_div8_fun(512))
        model.append(conv1)  # 0
        up1 = nn.Upsample(scale_factor=2)
        model.append(up1)  # 1
        cont1 = vn_layer.Concat()
        model.append(cont1)  # 2
        bsp1 = vn_layer.C3(make_div8_fun(512) + make_div8_fun(self.in_channels[0]), make_div8_fun(512),
                           make_round_fun(3), shortcut=False)
        model.append(bsp1)  # 3

        conv2 = vn_layer.Conv(make_div8_fun(512), make_div8_fun(256))
        model.append(conv2)  # 4
        up2 = nn.Upsample(scale_factor=2)
        model.append(up2)  # 5
        cont2 = vn_layer.Concat()
        model.append(cont2)  # 6
        bsp2 = vn_layer.C3(make_div8_fun(256) + make_div8_fun(self.in_channels[1]), make_div8_fun(256),
                           make_round_fun(3), shortcut=False)
        model.append(bsp2)  # 7

        conv3 = vn_layer.Conv(make_div8_fun(256), make_div8_fun(256), k=3, s=2)
        model.append(conv3)  # 8
        cont3 = vn_layer.Concat()
        model.append(cont3)  # 9
        bsp3 = vn_layer.C3(make_div8_fun(256) + make_div8_fun(256), make_div8_fun(512), make_round_fun(3),
                           shortcut=False)
        model.append(bsp3)  # 10

        conv4 = vn_layer.Conv(make_div8_fun(512), make_div8_fun(512), k=3, s=2)
        model.append(conv4)  # 11
        cont4 = vn_layer.Concat()
        model.append(cont4)  # 12
        bsp4 = vn_layer.C3(make_div8_fun(1024), make_div8_fun(1024), make_round_fun(3), shortcut=False)
        model.append(bsp4)  # 13

        self.det = nn.Sequential(*model)
        self.head = nn.Sequential(
            nn.Conv2d(make_div8_fun(256), 255, 1),
            nn.Conv2d(make_div8_fun(512), 255, 1),
            nn.Conv2d(make_div8_fun(1024), 255, 1),
        )

    def forward(self, feats):
        large_feat, inter_feat, small_feat = feats

        small_feat = self.det[0](small_feat)
        x = self.det[1](small_feat)
        x = self.det[2]([x, inter_feat])
        x = self.det[3](x)
        inter_feat = self.det[4](x)

        x = self.det[5](inter_feat)
        x = self.det[6]([x, large_feat])
        x = self.det[7](x)  # 128
        out0 = self.head[0](x)  # 第一个输出层

        x = self.det[8](x)
        x = self.det[9]([x, inter_feat])
        x = self.det[10](x)  #
        out1 = self.head[1](x)  # 第二个输出层

        x = self.det[11](x)
        x = self.det[12]([x, small_feat])
        x = self.det[13](x)  # 256
        out2 = self.head[2](x)  # 第三个输出层

        return tuple([out2, out1, out0]),  # 从小到大特征图返回

    @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.
        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            if 'pad_param' in img_metas[img_id]:
                pad_param = img_metas[img_id]['pad_param']
            else:
                pad_param = None
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, with_nms, pad_param)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single1(self,
                            pred_maps_list,
                            scale_factor,
                            cfg,
                            rescale=False,
                            with_nms=True,
                            pad_param=None):
        """Transform outputs for a single batch item into bbox predictions.
        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        nms_pre = 1000
        conf_thr = cfg.get('conf_thr', -1)

        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :4] = torch.sigmoid(pred_map[..., :4])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            if conf_thr > 0:
                conf_inds = conf_pred.ge(conf_thr).nonzero(
                    as_tuple=False).squeeze(1)
                bbox_pred = bbox_pred[conf_inds, :]
                cls_pred = cls_pred[conf_inds, :]
                conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        return self._bbox_post_process(multi_lvl_cls_scores, multi_lvl_bboxes,
                                       scale_factor, cfg, rescale, with_nms,
                                       multi_lvl_conf_scores, pad_param)

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           pad_param=None):
        """Transform outputs for a single batch item into bbox predictions.
        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels

        multi_pred_map = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map = torch.sigmoid(pred_map)
            pred_map[..., :4] = self.bbox_coder.decode(multi_lvl_anchors[i],
                                                       pred_map[..., :4], stride)
            multi_pred_map.append(pred_map)

        conf_thr = cfg.get('conf_thr', -1)

        mlvl_pred_map = torch.cat(multi_pred_map)
        if conf_thr > 0:
            conf_inds = mlvl_pred_map[..., 4].ge(conf_thr).nonzero(
                as_tuple=False).squeeze(1)
            mlvl_pred_map = mlvl_pred_map[conf_inds, :]

        if mlvl_pred_map.shape[0] == 0:
            return mlvl_pred_map[:, :4], mlvl_pred_map[:, 4]

        mlvl_pred_map[:, 5:] *= mlvl_pred_map[:, 4:5]  # conf = obj_conf * cls_conf

        conf, j = mlvl_pred_map[:, 5:].max(1, keepdim=True)
        mlvl_pred_map = torch.cat((mlvl_pred_map[:, :4], conf, j.float()), 1)
        if conf_thr > 0:
            mlvl_pred_map = mlvl_pred_map[conf.view(-1) > conf_thr, :]

        if mlvl_pred_map.shape[0] == 0:
            return mlvl_pred_map[:, :4], mlvl_pred_map[:, 4]
        elif mlvl_pred_map.shape[0] > 30000:  # excess boxes
            mlvl_pred_map = mlvl_pred_map[mlvl_pred_map[:, 4].argsort(descending=True)[:30000]]  # sort by confidence

        mlvl_bboxes = mlvl_pred_map[:, :4]
        if rescale:
            if pad_param is not None:
                mlvl_bboxes -= mlvl_bboxes.new_tensor(
                    [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = mlvl_pred_map[:, 4]
        mlvl_labels = mlvl_pred_map[:, 5]

        det_bboxes, keep = batched_nms(mlvl_bboxes, mlvl_scores.contiguous(), mlvl_labels, cfg.nms)
        return det_bboxes, mlvl_labels[keep]

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factor=None,
                           pad_param=None,
                           **kwargs):
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            if pad_param is not None:
                mlvl_bboxes -= mlvl_bboxes.new_tensor(
                    [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)

        if mlvl_score_factor is not None:
            mlvl_score_factor = torch.cat(mlvl_score_factor)

        if False:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                0,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_score_factor)
            return det_bboxes, det_labels
        else:
            if mlvl_score_factor is not None:
                return mlvl_bboxes, mlvl_scores, mlvl_score_factor
            else:
                return mlvl_bboxes, mlvl_scores
