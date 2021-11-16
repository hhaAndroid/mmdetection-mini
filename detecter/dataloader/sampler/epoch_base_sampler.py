import torch
import math
from torch.utils.data import DistributedSampler as _DistributedSampler
from ..builder import SAMPLER
from cvcore.utils import dist_comm as comm

__all__ = ['EpochBaseSampler']


@SAMPLER.register_module()
class EpochBaseSampler(_DistributedSampler):

    def __init__(self,
                 data_source,
                 shuffle=True,
                 seed=None):
        assert len(data_source) > 0
        if seed is None:
            seed = comm.shared_random_seed()
        self.seed = seed

        super().__init__(data_source,
                         num_replicas=comm.get_world_size(),
                         rank=comm.get_rank(),
                         shuffle=shuffle)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
