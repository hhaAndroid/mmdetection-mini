from cvcore.utils import dist_comm
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import copy
import operator

from ..dataset.dataset_wrappers import AspectRatioGroupedDataset
from cvcore import build_from_cfg
from .collate import trivial_batch_collator
from .builder import DATALOADER, SAMPLER

__all__ = ['build_default_dataloader', 'worker_init_fn']


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@DATALOADER.register_module()
def build_default_dataloader(dataset, samples_per_gpu, workers_per_gpu, sampler, batch_sampler=None,
                             collate=trivial_batch_collator, aspect_ratio_grouping=True, drop_last=True):
    if aspect_ratio_grouping is True:
        assert batch_sampler is None

    assert isinstance(sampler, dict)
    cp_sampler = copy.deepcopy(sampler)
    cp_sampler['data_source'] = dataset
    sampler = build_from_cfg(cp_sampler, SAMPLER)

    seed = None

    init_fn = partial(
        worker_init_fn, num_workers=workers_per_gpu, rank=dist_comm.get_rank(),
        seed=seed) if seed is not None else None

    if aspect_ratio_grouping:
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            num_workers=workers_per_gpu,
            collate_fn=operator.itemgetter(0),
            pin_memory=False,
            worker_init_fn=init_fn)
        return AspectRatioGroupedDataset(data_loader, samples_per_gpu)
    else:
        if batch_sampler is None:
            batch_sampler = BatchSampler(
                sampler, samples_per_gpu, drop_last=drop_last
            )
        else:
            batch_sampler = build_from_cfg(batch_sampler, SAMPLER)

        return DataLoader(
            dataset,
            num_workers=workers_per_gpu,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            worker_init_fn=init_fn,
        )
