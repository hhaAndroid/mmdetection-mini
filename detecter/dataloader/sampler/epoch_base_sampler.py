from torch.utils.data import DistributedSampler as _DistributedSampler
from ..builder import SAMPLER
from cvcore.utils import dist_comm as comm


__all__=['EpochBaseSampler']


@SAMPLER.register_module()
class EpochBaseSampler(_DistributedSampler):

    def __init__(self,
                 data_source,
                 drop_last=True,
                 shuffle=True,
                 seed=None):
        assert len(data_source)>0
        if seed is None:
            seed = comm.shared_random_seed()

        super().__init__(data_source,
                         num_replicas=comm.get_world_size(),
                         rank=comm.get_rank(),
                         drop_last=drop_last,
                         shuffle=shuffle,
                         seed=int(seed))
