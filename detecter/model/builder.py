from cvcore.utils import Registry, build_from_cfg

MODELS = Registry('models')

__all__ = ['BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES', 'DETECTORS',
           'build_backbone', 'build_neck', 'build_roi_extractor', 'build_shared_head', 'build_head',
           'build_loss', 'build_detector']

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
BRICKS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return build_from_cfg(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build_from_cfg(cfg, NECKS)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return build_from_cfg(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    """Build shared head."""
    return build_from_cfg(cfg, SHARED_HEADS)


def build_head(cfg):
    """Build head."""
    return build_from_cfg(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build_from_cfg(cfg, LOSSES)


def build_detector(cfg, default_args=None):
    """Build detector."""
    return build_from_cfg(cfg, DETECTORS, default_args)


def build_bricks(cfg):
    """Build detector."""
    return build_from_cfg(cfg, DETECTORS)