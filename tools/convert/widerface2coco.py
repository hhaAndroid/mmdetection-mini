import os
import os.path as osp
import argparse
from PIL import Image

import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

from mmdet import cv_core

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert WIDER FACE annotations to coco format')
    parser.add_argument('gt_txt_path', help='gt text path')
    parser.add_argument('images_path', help='images path')
    parser.add_argument('-o', '--out-dir', default='./', help='output path')
    parser.add_argument('-s', '--save_name', default='temp.json', help='output file name')
    args = parser.parse_args()
    return args


# wider face数据集格式说明
# http://shuoyang1213.me/WIDERFACE/
# File name
# Number of bounding box
# x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

def main():
    args = parse_args()
    gt_txt_path = args.gt_txt_path
    root_img_path = args.images_path
    out_dir = args.out_dir
    save_name = args.save_name
    cv_core.mkdir_or_exist(os.path.join(out_dir, 'annotations'))

    # 手动定义
    categories = [
        {
            'id': 1,  # 只有一个类别
            'name': 'face',
            'supercategory': 'object',
        }
    ]
    coco_creater = cv_core.CocoCreator(categories, out_dir=out_dir, save_name=save_name)

    with open(gt_txt_path) as f:  # gt txt路径
        image_id = 1  # 手动设置
        segmentation_id = 1
        while True:
            imFilename = f.readline().strip()  # 读取第一行，例如 0--Parade/0_Parade_marchingband_1_849.jpg
            if imFilename:
                print(imFilename)
                img_path = os.path.join(root_img_path, imFilename)
                image = Image.open(img_path)
                # 第一步
                coco_creater.create_image_info(image_id, imFilename, image.size)
                class_id = categories[0]['id']  # 只有一个类别
                category_info = {'id': class_id, 'is_crowd': 0}

                nbBndboxes = f.readline()  # 读取第二行，例如1
                if int(nbBndboxes) == 0:
                    _ = f.readline()
                else:
                    i = 0
                    while i < int(nbBndboxes):  # 遍历读取每个bbox
                        i = i + 1
                        x1, y1, w, h, _, _, _, _, _, _ = [int(i) for i in f.readline().split()]
                        if w < 0 or h < 0:
                            continue
                        # 第二步
                        coco_creater.create_annotation_info(segmentation_id, image_id, category_info,
                                                            bounding_box=[x1, y1, w, h])
                        segmentation_id += 1
                    image_id += 1
            else:
                break
    # 第三步
    coco_creater.dump()


# python widerface2coco.py /home/pi/dataset/wider_face/wider_face_split/wider_face_train_bbx_gt.txt \
# /home/pi/dataset/wider_face/WIDER_train/images
if __name__ == '__main__':
    main()
