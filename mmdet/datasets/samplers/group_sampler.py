from __future__ import division

import numpy as np
from torch.utils.data import Sampler


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)  # 所有图片的flag标志
        self.group_sizes = np.bincount(self.flag)  # 两组，分别统计两组的图片个数
        self.num_samples = 0
        # 考虑分组batch情况下的样本总数,注意是向上取整
        for i, size in enumerate(self.group_sizes):
            # 例如如果某组是1000样本，batch=9,那么本代码实际迭代了112个batch，而不是111
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        # 构造迭代器，核心就是构建self.num_samples个样本index
        # 前n个是第一个分组，后m个是第二个分组，这样后面pad的时候就不会出现一大块黑边
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            # 由于是向上取整，为了保证和self.num_samples个数一样多，不够的就随机插入
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        # 虽然这里有打乱操作(连续切割模式)，但由于前面每组index都是正好够batch的，所以也不会出现某个batch中同时含有两种flag的数据
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples
