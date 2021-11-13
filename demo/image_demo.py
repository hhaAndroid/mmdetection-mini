# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import cvcore
from detecter.model import build_detector
from detecter.utils import DetectionCheckpointer
import torch
from detecter.dataset import Compose
from loguru import logger
import mmcv
import cv2


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    config = cvcore.Config.fromfile(config)
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model)
    if checkpoint is not None:
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(checkpoint)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    cfg = model.cfg
    test_pipeline = Compose(cfg.data.test.pipeline)
    img = test_pipeline(dict(img_info=dict(filename=img), img_prefix=None))
    # forward the model
    with torch.no_grad():
        results = model([img])
    return results


def show_bbox(img, results):
    instances = results[0]
    instance = instances.to('cpu')
    pred_boxes = instance.bboxes.tensor.detach()
    print(pred_boxes)
    pred_scores = instance.scores
    dets = torch.cat([pred_boxes, pred_scores[:, None]], -1).numpy()
    result_img = mmcv.imshow_det_bboxes(img, dets, instance.labels, score_thr=0, show=False, thickness=4)
    cv2.namedWindow('bbox', 0)
    cv2.imshow('bbox', result_img)
    cv2.waitKey(0)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='../configs/yolov5/yolov5s_v6.py',help='Config file')
    parser.add_argument('--checkpoint', default='../yolov5s_mm_new.pth',help='Checkpoint file')
    parser.add_argument('--img', default='zidane.jpg',
                        help='Image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


@logger.catch
def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    show_bbox(args.img, result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
