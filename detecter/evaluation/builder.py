from cvcore import Registry, build_from_cfg, Logger
from collections.abc import Mapping
from contextlib import contextmanager


__all__ = ['build_evaluator', 'EVALUATORS', 'print_csv_format', 'inference_context']

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
