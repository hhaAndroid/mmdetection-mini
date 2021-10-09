from .base_evaluator import DatasetEvaluator
from .builder import EVALUATORS
from cvcore import build_from_cfg
from collections import OrderedDict
from cvcore.utils import is_main_process

__all__ = ['ListDatasetEvaluator', 'CompositeDatasetEvaluator']


@EVALUATORS.register_module()
class ListDatasetEvaluator(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        self._evaluators = [build_from_cfg(evaluator, EVALUATORS) for evaluator in evaluators]

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                            k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


# 用于在不同时刻采用不同评估器策略
@EVALUATORS.register_module()
class CompositeDatasetEvaluator(DatasetEvaluator):
    pass