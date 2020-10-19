import numpy as np
from multiprocessing import Pool
from ..bbox import bbox_overlaps


# https://zhuanlan.zhihu.com/p/34655990
def calc_PR_curve(pred, label):
    pos = label[label == 1]  # 正样本
    threshold = np.sort(pred)[::-1]  # pred是每个样本的正例预测概率值,逆序
    label = label[pred.argsort()[::-1]]
    precision = []
    recall = []
    tp = 0
    fp = 0
    ap = 0  # 平均精度
    for i in range(len(threshold)):
        if label[i] == 1:
            tp += 1
            recall.append(tp / len(pos))
            precision.append(tp / (tp + fp))
            # 近似曲线下面积
            ap += (recall[i] - recall[i - 1]) * precision[i]
        else:
            fp += 1
            recall.append(tp / len(pos))
            precision.append(tp / (tp + fp))

    return precision, recall, ap


def tpfp_voc(det_bboxes, gt_bboxes, iou_thr=0.5):
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]

    # tp和fp都是针对预测个数而言，不是gt个数
    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)

    # 如果gt=0，那么所有预测框都算误报，所有预测bbox位置的fp都设置为1
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp

    if num_dets == 0:
        return tp, fp

    ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes).numpy()
    # print(ious)
    # 对于每个预测框，找到最匹配的gt iou
    ious_max = ious.max(axis=1)
    # 对于每个预测框，找到最匹配gt的索引
    ious_argmax = ious.argmax(axis=1)

    # 按照预测概率分支降序排列
    sort_inds = np.argsort(-det_bboxes[:, -1])
    gt_covered = np.zeros(num_gts, dtype=bool)
    # 多对一情况下，除了概率分值最大且大于阈值的预测框算tp外，其他框全部算fp
    for i in sort_inds:
        # 如果大于iou，则表示匹配
        if ious_max[i] >= iou_thr:
            matched_gt = ious_argmax[i]
            # 每个gt bbox只匹配一次，且是和预测概率最大的匹配，不是按照iou
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def _average_precision(recalls, precisions, mode='voc2007'):
    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'voc2012':  # 平滑后就是标准的PR曲线算法
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        # 写法比较高级，高效
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])  # 每段区间内，精度都是取最大值，也就是水平线
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]  # 找到召回率转折点，表示x轴移动区间索引
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])  # 每段面积和
    elif mode == 'voc2007':  # 11点法，需要平平滑处理
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


# code ref from mmdetection
def voc_eval_map(results, annotations, iou_thr=0.5, name='voc2007', nproc=4):
    """
    :param results: list[list],外层list是指代图片编号，内层list是指代类别编号，
    假设一共20个类，则内层list长度为20，每个List内部是numpy矩阵，nx5表示每张图片对应的每个类别的检测bbox，xyxyconf格式
    :param annotations:和results一样
    :param iou_thr: 是否算TP的阈值，voc默认是0.5
    :param name: 采用哪一种评估指标，voc2007是11点，voc2012是标准pr曲线计算
    :return:
    """
    assert len(results) == len(annotations)
    num_imgs = len(results)  # 图片个数
    num_classes = len(results[0])  # positive class num
    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        cls_dets = [img_res[i] for img_res in results]
        cls_gts = [img_res[i] for img_res in annotations]
        tpfp = pool.starmap(
            tpfp_voc,
            zip(cls_dets, cls_gts, [iou_thr for _ in range(num_imgs)]))
        # 得到每个预测bbox的tp和fp情况
        tp, fp = tuple(zip(*tpfp))
        # 统计gt bbox数目
        num_gts = 0
        for j, bbox in enumerate(cls_gts):
            num_gts += bbox.shape[0]

        # 合并所有图片所有预测bbox
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]  # 检测bbox个数

        # 以上计算出了每个预测bbox的tp和fp情况
        # 此处计算精度和召回率，写的比较高级
        sort_inds = np.argsort(-cls_dets[:, -1])  # 按照预测概率分值降序排列
        # 仔细思考这种写法，其实是c3_pr_roc.py里面calc_PR_curve的高级快速写法
        tp = np.hstack(tp)[sort_inds][None]
        fp = np.hstack(fp)[sort_inds][None]
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
        recalls = recalls[0, :]
        precisions = precisions[0, :]
        # print('recalls', recalls, 'precisions', precisions)
        ap = _average_precision(recalls, precisions, name)[0]
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0
    return mean_ap
