import torch
import numpy as np
import cv2
import mmdet.cv_core as mmcv
import matplotlib.pyplot as plt


def get_target_mask(gt_bboxes, feature_shape, center_sample_radius, center_sampling):
    # 得到points
    xs = torch.arange(0, feature_shape[1])
    ys = torch.arange(0, feature_shape[0])
    y, x = torch.meshgrid(ys, xs)  # 注意，返回的一定是y在前
    y = y.flatten()  # hw
    x = x.flatten()  # hw
    # 还原到原图
    # points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
    #                      dim=-1) + stride // 2  # 整体偏移stride//2，对应中心点
    # 我们假设就是原图
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)

    num_points = points.size(0)  # 100x100,2
    num_gts = gt_bboxes.size(0)  # 1x4
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)  # 100x100,1,4
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_gts)  # 100x100,1
    ys = ys[:, None].expand(num_points, num_gts)

    if center_sampling:
        center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        # center_gts里面存储的相当于是新的缩放后bbox坐标了
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_ones(center_xs.shape) * center_sample_radius
        x_mins = center_xs - stride
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride
        # 如果stride值比较小，x_mins还在bbox内部，则不做处理
        # 如果stride值比较大，x_mins已经出bbox界限了，则强制规定x_mins=gt_bboxes[..., 0]，相当于center_sampling无效
        center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                         x_mins, gt_bboxes[..., 0])
        center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                         y_mins, gt_bboxes[..., 1])
        center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                         gt_bboxes[..., 2], x_maxs)
        center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                         gt_bboxes[..., 3], y_maxs)
    else:
        center_gts = gt_bboxes

    # 计算原图上面任意一点距离bbox4条边的距离
    left = xs - center_gts[..., 0]  # 特征图上面点距离bbox左边界距离
    right = center_gts[..., 2] - xs  # 注意谁减谁
    top = ys - center_gts[..., 1]
    bottom = center_gts[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)  # 100x100,1,4
    # value, index = bbox_targets.min(-1)
    pos_mask = bbox_targets.min(-1)[0] > 0
    pos_mask = pos_mask.view(feature_shape[0], feature_shape[1], -1)
    return pos_mask, bbox_targets


def centerness_target(pos_mask, bbox_targets):
    """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
    # only calculate pos centerness targets, otherwise there may be nan
    pos_mask = pos_mask.view(-1, 1)
    bbox_targets = bbox_targets[pos_mask]
    left_right = bbox_targets[:, [0, 2]]
    top_bottom = bbox_targets[:, [1, 3]]
    centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                         (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    targets = torch.sqrt(centerness_targets)
    # 还原成图返回
    img_disp_target = pos_mask.new_zeros(pos_mask.shape, dtype=torch.float32)
    img_disp_target[pos_mask] = targets
    return img_disp_target


if __name__ == '__main__':
    # 缺点： 中心采样策略无法反映hw变化，而且既然叫做半径，为啥mask区域不是圆形，而是正方形
    center_sampling = True  # 是否使用中心采样策略
    feature_shape = (100, 100, 3)
    strides = 4
    radius = 3.5  # 默认1.5
    center_sample_radius = radius * strides  # 扩展半径radius，值越大，扩展面积越大
    gt_boox = [20, 30, 80, 71]  # 特征图size xyxy

    gt_bbox = torch.as_tensor(gt_boox, dtype=torch.float32).view(-1, 4)
    pos_mask, bbox_targets = get_target_mask(gt_bbox, feature_shape, center_sample_radius, center_sampling)

    # 可视化
    pos_mask1 = pos_mask[..., 0].numpy()
    gray_img = np.where(pos_mask1 > 0, 255, 0).astype(np.uint8)
    # 绘制原始bbox
    img = mmcv.gray2bgr(gray_img)
    cv2.rectangle(img, (gt_boox[0], gt_boox[1]), (gt_boox[2], gt_boox[3]), color=(255, 0, 0))
    cv2.namedWindow('img', 0)
    mmcv.imshow(img, 'img')

    # 显示centerness
    centerness_targets = centerness_target(pos_mask, bbox_targets)
    centerness_targets = centerness_targets.view(feature_shape[0], feature_shape[1])
    plt.imshow(centerness_targets)
    plt.show()
