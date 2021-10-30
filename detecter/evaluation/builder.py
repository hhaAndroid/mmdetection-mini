from cvcore import Registry, build_from_cfg, Logger, get_event_storage
from .base_evaluator import DatasetEvaluator
from ..dataset import build_dataset
from ..dataloader import build_dataloader
import copy
from collections import OrderedDict
from collections.abc import Mapping
from cvcore.utils import dist_comm
import time
import torch
from contextlib import ExitStack, contextmanager
import datetime
import torch.nn as nn

__all__ = ['build_evaluator', 'EVALUATORS', 'eval_func', 'inference_on_dataset', 'print_csv_format']

EVALUATORS = Registry('evaluator')


def build_evaluator(cfg, default_args=None):
    dataset = build_from_cfg(cfg, EVALUATORS, default_args)
    return dataset


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            Logger.info("copypaste: Task: {}".format(task))
            Logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            Logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            Logger.info(f"copypaste: {task}={res}")


def eval_func(global_cfg, model):
    """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
    if 'evaluator' not in global_cfg:
        raise NotImplementedError

    val_dataset = build_dataset(global_cfg.data.val)
    val_dataloader = build_dataloader(global_cfg.dataloader.val, val_dataset)

    cp_evaluator_cfg = copy.deepcopy(global_cfg.evaluator.eval_func)
    cp_evaluator_cfg['dataloader'] = val_dataloader

    evaluator = build_from_cfg(cp_evaluator_cfg, EVALUATORS)
    assert isinstance(evaluator, DatasetEvaluator)

    results = inference_on_dataset(model, val_dataloader, evaluator)
    if dist_comm.is_main_process():
        assert isinstance(
            results, dict
        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
            results
        )
        # Logger.info("Evaluation results for {} in csv format:".format(val_dataset.__name__))
        print_csv_format(results)

    return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = dist_comm.get_world_size()
    Logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    eval_log_interv = 50

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
            if idx % eval_log_interv == 0:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                Logger.info(
                    f"Inference done {idx + 1}/{total}. "
                    f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                    f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                    f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                    f"Total: {total_seconds_per_iter:.4f} s/iter. "
                    f"ETA={eta}"
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    Logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    Logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
