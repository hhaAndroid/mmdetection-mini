import argparse
import os.path as osp
import numpy as np
import shutil

from pycocotools.coco import COCO
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='extract coco subset')
    parser.add_argument('--root', default='/home/PJLAB/huanghaian/dataset/coco', help='root path')
    parser.add_argument(
        '--output-dir',
        default='/home/PJLAB/huanghaian/dataset/subsetcoco',
        type=str,
        help='save root dir')
    parser.add_argument('--num-img', default=10, help='num of extract image')
    args = parser.parse_args()
    return args


def _process_data(args, type):
    if type == 'train':
        ann_file_name = 'annotations/instances_train2017.json'
    else:
        ann_file_name = 'annotations/instances_val2017.json'
    ann_path = osp.join(args.root, ann_file_name)
    json_data = mmcv.load(ann_path)

    new_json_data = {'info': json_data['info'], 'licenses': json_data['licenses'],
                     'categories': json_data['categories'], 'images': [], 'annotations': []}

    images = json_data['images']
    coco = COCO(ann_path)

    # shuffle
    np.random.shuffle(images)

    progress_bar = mmcv.ProgressBar(args.num_img)
    for i in range(args.num_img):
        file_name = images[i]['file_name']
        stuff_file_name = osp.splitext(file_name)[0] + '.png'
        image_path = osp.join(args.root, type + '2017', file_name)
        stuff_image_path = osp.join(args.root, 'stuffthingmaps', type + '2017', stuff_file_name)
        ann_ids = coco.getAnnIds(imgIds=[images[i]['id']])
        ann_info = coco.loadAnns(ann_ids)
        new_json_data['images'].append(images[i])
        new_json_data['annotations'].extend(ann_info)

        shutil.copy(image_path, osp.join(args.output_dir, type + '2017'))
        if osp.exists(stuff_image_path):
            shutil.copy(stuff_image_path, osp.join(args.output_dir, 'stuffthingmaps', type + '2017'))

        progress_bar.update()

    mmcv.dump(new_json_data, osp.join(args.output_dir, ann_file_name))


def _make_dir(output_dir):
    mmcv.mkdir_or_exist(output_dir)
    mmcv.mkdir_or_exist(osp.join(output_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'val2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/val2017'))


def main():
    args = parse_args()
    assert args.output_dir != args.root

    _make_dir(args.output_dir)
    _process_data(args, 'train')
    _process_data(args, 'val')


if __name__ == '__main__':
    main()
