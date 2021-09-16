import itertools
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler
from cvcore.utils import dist_comm as comm
from ..builder import SAMPLER

__all__ = ['InfiniteSampler']


@SAMPLER.register_module()
class InfiniteSampler(Sampler):
    def __init__(self, data_source, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        super().__init__(data_source)
        self._size = len(data_source)
        assert self._size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()
