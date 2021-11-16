from torch.utils.data.sampler import Sampler
from cvcore.utils import dist_comm as comm
from ..builder import SAMPLER

__all__ = ['InferenceSampler']


@SAMPLER.register_module()
class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, data_source, **kwargs):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        super().__init__(data_source)
        self._size = len(data_source)
        assert self._size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
