from .coco import CocoDataset


class WiderFace_CocoFormat(CocoDataset):
    CLASSES = ('face', )