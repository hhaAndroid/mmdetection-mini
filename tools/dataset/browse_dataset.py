import argparse
from cvcore import DictAction, Config, convert_image_to_rgb, ColorMode
from detecter.dataset import build_dataset
from detecter.dataloader import build_dataloader
from detecter.visualizer import DetVisualizer
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--dataloader', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=5,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(eval(f'cfg.data.{args.mode}'))
    gloabl_metas = dataset.get_global_metas()
    loader = dataset

    if args.dataloader:
        dataloader_cfg = eval(f'cfg.dataloader.{args.mode}')
        dataloader_cfg['samples_per_gpu'] = 1
        dataloader_cfg['workers_per_gpu'] = 0
        dataloader = build_dataloader(dataloader_cfg, dataset)
        loader = dataloader

    visualizer = DetVisualizer(metadata=None)
    for data in loader:
        if not args.dataloader:
            data = [data]
        img = data[0]['img']
        data_sample = data[0]['data_sample']

        visualizer.set_image(convert_image_to_rgb(img.permute(1,2,0), "RGB"))
        visualizer.draw(data_sample)
        gt_draw_image=visualizer.get_image()

        # 假装有pred
        gt_bboxes = data_sample.gt_instances.bboxes.clone().tensor
        gt_bboxes = gt_bboxes + 10
        from detecter.core.structures import Boxes, InstanceData
        pred_instance = InstanceData()
        pred_instance.bboxes = Boxes(gt_bboxes)
        pred_instance.labels = data_sample.gt_instances.labels
        data_sample.pred_instances = pred_instance
        data_sample.pop('gt_instances')
        visualizer.set_image(convert_image_to_rgb(img.permute(1, 2, 0), "RGB"))
        visualizer.draw(data_sample)
        pred_draw_image = visualizer.get_image()
        import numpy as np
        concat = np.concatenate((gt_draw_image, pred_draw_image), axis=1)

        visualizer.show(concat)


if __name__ == '__main__':
    main()
