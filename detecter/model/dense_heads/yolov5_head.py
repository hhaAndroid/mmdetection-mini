# -*- coding:utf-8 -*-
import functools
import math

import torch
import torch.nn as nn

from mmcv.runner import get_dist_info

from detecter.core.structures import InstanceData, Boxes

from .anchor_head import AnchorHead
from ..builder import HEADS
from ..utils import brick as vn_layer


__all__ = ['YOLOV5Head']


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(_make_divisible, divisor=divisor, width_multiple=width_multiple)


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


@HEADS.register_module()
class YOLOV5Head(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOV5BBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 **kwargs):

        self.featmap_strides=featmap_strides
        self.out_channels=out_channels

        super(YOLOV5Head, self).__init__(num_classes,in_channels,out_channels,anchor_generator,bbox_coder,**kwargs)
        self.loss_fun = ComputeLoss(self, self.anchor_generator)



    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes


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
            nn.Conv2d(make_div8_fun(256), (5 + self.num_classes) * self.num_anchors, 1),
            nn.Conv2d(make_div8_fun(512), (5 + self.num_classes) * self.num_anchors, 1),
            nn.Conv2d(make_div8_fun(1024), (5 + self.num_classes) * self.num_anchors, 1),
        )

    def forward_layer(self, features, batched_inputs=None):
        large_feat, inter_feat, small_feat = features

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

    def init_weights(self):

        def _initialize_biases(model, stride=[8, 16, 32],
                               cf=None):  # initialize biases into Detect(), cf is class frequency
            # https://arxiv.org/abs/1708.02002 section 3.3
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
            for mi, s in zip(model, stride):  # from
                b = mi.bias.view(3, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999)) if cf is None else torch.log(
                    cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        _initialize_biases(self.head)

        # for test
        # from mmcv.cnn import constant_init
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         constant_init(m, 1)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         constant_init(m, 1)

    def loss(self,
             pred_maps,
             batched_inputs,
             **kwargs):

        device = pred_maps[0].device

        gt_instances = [data['data_sample'].gt_instances.to(device) for data in batched_inputs]
        gt_bboxes_list = [instance.bboxes.tensor for instance in gt_instances]
        gt_labels_list = [instance.labels for instance in gt_instances]
        img_metas = [x["img_metas"] for x in batched_inputs]

        loss = self.loss_fun(pred_maps, gt_bboxes_list, gt_labels_list, img_metas)
        return loss


    def get_bboxes(self,
                   pred_maps,
                   batched_inputs,
                   skip_post=False,
                   **kwargs):

        img_metas = [x["img_metas"] for x in batched_inputs]

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
                                                self.test_cfg, pad_param)
            result_list.append(proposals)
        return result_list


    def _get_bboxes_single(self,
                              pred_maps_list,
                              scale_factor,
                              cfg,
                              pad_param=None):

        multi_pred_map = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)

        for i in range(num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            # pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)
            # 是否有 contiguous 对数值有影响
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib).contiguous()

            pred_map = torch.sigmoid(pred_map)
            pred_map[..., :4] = self.bbox_coder.decode(multi_lvl_anchors[i],
                                                       pred_map[..., :4], stride)
            multi_pred_map.append(pred_map)

        conf_thr = cfg.get('conf_thr', -1)
        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        max_pre_nms = 30000

        mlvl_pred_map = torch.cat(multi_pred_map)
        if conf_thr > 0:
            conf_inds = mlvl_pred_map[..., 4].ge(conf_thr).nonzero(
                as_tuple=False).squeeze(1)
            mlvl_pred_map = mlvl_pred_map[conf_inds, :]

        if mlvl_pred_map.shape[0] == 0:
            return mlvl_pred_map[:, :4], mlvl_pred_map[:, 4]

        mlvl_pred_map[:, 5:] *= mlvl_pred_map[:, 4:5]  # conf = obj_conf * cls_conf

        if multi_label:
            i, j = (mlvl_pred_map[:, 5:] > conf_thr).nonzero(as_tuple=False).T
            mlvl_pred_map = torch.cat((mlvl_pred_map[:, :4][i], mlvl_pred_map[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = mlvl_pred_map[:, 5:].max(1, keepdim=True)
            mlvl_pred_map = torch.cat((mlvl_pred_map[:, :4], conf, j.float()), 1)
            mlvl_pred_map = mlvl_pred_map[conf.view(-1) > conf_thr, :]

        if mlvl_pred_map.shape[0] == 0:
            return mlvl_pred_map[:, :4], mlvl_pred_map[:, 4]
        elif mlvl_pred_map.shape[0] > max_pre_nms:  # excess boxes
            mlvl_pred_map = mlvl_pred_map[mlvl_pred_map[:, 4].argsort(descending=True)[:max_pre_nms]]

        mlvl_bboxes = mlvl_pred_map[:, :4]

        if pad_param is not None:
            mlvl_bboxes -= mlvl_bboxes.new_tensor(
                    [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)


        c = mlvl_pred_map[:, 5:6] * 4096  # classes
        boxes, scores = mlvl_pred_map[:, :4] + c, mlvl_pred_map[:, 4]  # boxes (offset by class), scores
        import torchvision
        i = torchvision.ops.nms(boxes, scores, cfg.nms.iou_threshold)  # NMS
        if i.shape[0] > 300:  # limit detections
            i = i[:300]

        det_bboxes = mlvl_pred_map[:, :5][i]
        det_label = mlvl_pred_map[:, 5][i]


        pred_result = InstanceData()
        pred_result.bboxes = Boxes(det_bboxes[:,:4])
        pred_result.scores = det_bboxes[:, 4]
        pred_result.labels = det_label

        return pred_result



class ComputeLoss:
    # Compute losses
    def __init__(self, model, anchor_generator, autobalance=False):
        super(ComputeLoss, self).__init__()
        h = {'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
             'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0,
             'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
             'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
             'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'label_smoothing': 0.0}

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']])).cuda()
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']])).cuda()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = 1.0, 0.0  # positive, negative BCE targets

        self.balance = [4.0, 1.0, 0.4]
        self.ssi = 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        self.na = 3
        self.nl = 3
        self.nc = model.num_classes
        self.no = model.num_classes + 5

        # 暂时调整位置
        base_sizes = anchor_generator.base_sizes[::-1]
        strides = anchor_generator.strides[::-1]
        self.anchors = torch.tensor(base_sizes).float().view(self.nl, -1, 2)
        self.stride = torch.tensor(strides).float().view(self.nl, -1, 2)
        # 除以 stride
        self.anchors /= self.stride
        self.anchors = self.anchors.cuda()

    def __call__(self, pred_maps, gt_bboxes, gt_labels, img_metas):  # predictions, targets, model
        # 暂时调整位置
        pred_maps = pred_maps[::-1]
        device = pred_maps[0].device
        p = []
        for i in range(3):
            bs, _, ny, nx = pred_maps[i].shape
            p.append(pred_maps[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, gt_bboxes, gt_labels, img_metas)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        # loss = lbox + lobj + lcls
        # print(loss.item(), lbox.item(), lobj.item(), lcls.item())
        _, world_size = get_dist_info()
        return dict(
            loss_cls=lcls * bs * world_size,
            loss_conf=lobj * bs * world_size,
            loss_bbox=lbox * bs * world_size)

    def build_targets(self, p, gt_bboxes, gt_labels, img_metas):
        # bs
        target_list = []
        for i in range(len(gt_bboxes)):
            # 归一化且变成 cxcywh 格式
            img_shape = img_metas[i]['batch_input_shape']
            bbox = gt_bboxes[i].clone()
            bbox[:, 0] = (gt_bboxes[i][:, 0] + gt_bboxes[i][:, 2]) / 2  # x center
            bbox[:, 1] = (gt_bboxes[i][:, 1] + gt_bboxes[i][:, 3]) / 2  # y center
            bbox[:, 2] = gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]  # width
            bbox[:, 3] = gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]  # height
            bbox[:, 1::2] = bbox[:, 1::2] / (img_shape[0] * 1.0)  # normalized height 0-1
            bbox[:, 0::2] = bbox[:, 0::2] / (img_shape[1] * 1.0)  # normalized width 0-1

            index = gt_labels[i].new_full((len(gt_bboxes[i]),), i)
            target = torch.cat((index[:, None].float(), gt_labels[i][:, None].float(), bbox.float()), dim=1)
            target_list.append(target)

        targets = torch.cat(target_list, dim=0)

        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
