from .builder import POST_PROCESSORS

__all__ = ['ResizeResultsToOri']


@POST_PROCESSORS.register_module()
class ResizeResultsToOri(object):
    """Mapping results to the scale of original image.
    """
    def __call__(self, results, image_meta=None):
        scale_factor = image_meta.get('scale_factor', 1.0)
        results.bboxes.scale(1.0 / scale_factor[0], 1.0 / scale_factor[1])
        return results
