# Copyright (c) Open-MMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from .data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmdet.cv_core.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        # 之所以有这个设定，是为了对付多gpu训练时候
        # 假设一共4张卡，每张卡8个样本，故dataloader吐出来的batch=32
        # 但是分组采样时候是对单batch而言的，也就是说这里收集到的32个batch其实分成了4组
        # 4组里面可能存在flag不一样的组，如果这里不对每组进行单独操作
        # 那么其实前面写的分组采样功能就没多大用途了。本函数写法会出现4个组输出的shape不一样，但是由于是分配到4块卡上训练，所以大小不一样也没有关系
        # 保持单张卡内shape一样就行。
        # 所以对于单卡训练场景，len(batch)=samples_per_gpu，这里的for循环没有意义,可以删掉

        # 如果要兼容batch test，那么这个就不能要，因为最后一个batch可能不是整数倍
        # assert len(batch) % samples_per_gpu == 0

        stacked = []
        if batch[0].cpu_only:  # meta data 直接变成list，然后dc
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:  # img tensor 先全部统一大小，然后stack，最后包裹为dc
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        # 由于前面pipeline没有保证batch内图片大小一致，故在这里需要强制pad到最大shape
                        # 也是向后右pad
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))  # 如果是分布式多卡训练，可能每个组的shape是不一样的，没有影响
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:  # 其余信息，例如gt bbox，内部是tensor,变成list[tensor]，然后dc
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)  # 输出的所有对象都是用dc包裹了
