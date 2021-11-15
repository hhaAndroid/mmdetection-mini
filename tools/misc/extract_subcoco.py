import argparse
import os.path as osp
import numpy as np
import shutil

from pycocotools.coco import COCO
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='extract coco subset')
    parser.add_argument('--root', default='/home/hha/dataset/coco', help='root path')
    parser.add_argument(
        '--output-dir',
        default='/home/hha/dataset/subsetcoco200',
        type=str,
        help='save root dir')
    parser.add_argument('--num-img', default=200, help='num of extract image')
    args = parser.parse_args()
    return args


def _process_data(args, type):
    if type == 'train':
        ann_file_name = 'annotations/instances_train2017.json'
        yolov5_label_name='train2017.txt'
    else:
        ann_file_name = 'annotations/instances_val2017.json'
        yolov5_label_name = 'val2017.txt'
    yolov5_label_name=osp.join(args.output_dir, yolov5_label_name)
    ann_path = osp.join(args.root, ann_file_name)
    json_data = mmcv.load(ann_path)

    new_json_data = {'info': json_data['info'], 'licenses': json_data['licenses'],
                     'categories': json_data['categories'], 'images': [], 'annotations': []}

    images = json_data['images']
    coco = COCO(ann_path)

    # shuffle
    np.random.shuffle(images)

    label_file=open(yolov5_label_name,'w')

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

        image_path = osp.join('./'+type + '2017', file_name)
        label_file.write(image_path+'\n')

        progress_bar.update()

    mmcv.dump(new_json_data, osp.join(args.output_dir, ann_file_name))
    label_file.close()


def _make_dir(output_dir):
    mmcv.mkdir_or_exist(output_dir)
    mmcv.mkdir_or_exist(osp.join(output_dir, 'annotations'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'val2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/train2017'))
    mmcv.mkdir_or_exist(osp.join(output_dir, 'stuffthingmaps/val2017'))


import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x


def convert_coco_json(json_file, out,use_segments=False, cls91to80=True):
    coco80 = coco91_to_coco80_class()

    with open(json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['iscrowd']:
            continue

        img = images['%g' % x['image_id']]
        h, w, f = img['height'], img['width'], img['file_name']

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Segments
        if use_segments:
                segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
                s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
                line = cls, *(s if use_segments else box)  # cls, box or segments
                with open((Path(out) / f).with_suffix('.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


def main():
    args = parse_args()
    assert args.output_dir != args.root

    _make_dir(args.output_dir)
    _process_data(args, 'train')
    _process_data(args, 'val')

    convert_coco_json(f'{args.output_dir}/annotations/instances_train2017.json',f'{args.output_dir}/train2017')
    convert_coco_json(f'{args.output_dir}/annotations/instances_val2017.json',f'{args.output_dir}/val2017')


if __name__ == '__main__':
    main()
