from cvcore import Registry, build_from_cfg

__all__ = ['build_evaluator', 'EVALUATORS']

EVALUATORS = Registry('evaluator')


def build_evaluator(cfg, default_args=None):
    dataset = build_from_cfg(cfg, EVALUATORS, default_args)
    return dataset
