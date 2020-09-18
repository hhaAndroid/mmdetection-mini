# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def showBBox(coco, anns, label_box=True, is_filling=True):
    """
    show bounding box of annotations or predictions
    anns: loadAnns() annotations or predictions subject to coco results format
    label_box: show background of category labels or not
    """
    if len(anns) == 0:
        return 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
    for ann in anns:
        c = image2color[ann['category_id']]
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)
        if label_box:
            label_bbox = dict(facecolor=c)
        else:
            label_bbox = None
        if 'score' in ann:
            ax.text(bbox_x, bbox_y, '%s: %.2f' % (coco.loadCats(ann['category_id'])[0]['name'], ann['score']),
                    color='white', bbox=label_bbox)
        else:
            ax.text(bbox_x, bbox_y, '%s' % (coco.loadCats(ann['category_id'])[0]['name']), color='white',
                    bbox=label_bbox)
    if is_filling:
        # option for filling bounding box
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


# only_bbox 为True表示仅仅可视化bbox，其余label不显示
# show_all 表示所有类别都显示，否则category_name来确定显示类别
def show_coco(data_root, ann_file, img_prefix, only_bbox=True, show_all=True, category_name='bicycle'):
    example_coco = COCO(ann_file)
    print('图片总数：{}'.format(len(example_coco.getImgIds())))
    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

    if show_all:
        category_ids = []
    else:
        category_ids = example_coco.getCatIds(category_name)
    image_ids = example_coco.getImgIds(catIds=category_ids)

    for i in range(len(image_ids)):
        plt.figure()
        image_data = example_coco.loadImgs(image_ids[i])[0]
        path = os.path.join(data_root, img_prefix, image_data['file_name'])
        image = cv2.imread(path)
        plt.imshow(image)
        annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
        annotations = example_coco.loadAnns(annotation_ids)
        if only_bbox:
            showBBox(example_coco, annotations)
        else:
            example_coco.showAnns(annotations)
        plt.show()


if __name__ == '__main__':
    # 和cfg里面设置一样 coco
    data_root = '/home/pi/dataset/coco/'
    ann_file = data_root + 'annotations/instances_val2017.json'
    img_prefix = data_root + 'images/val2017/'
    show_coco(data_root, ann_file, img_prefix)

    # voc转化为coco后显示
    data_root = '/home/pi/dataset/VOCdevkit/'
    ann_file = data_root + 'annotations/voc0712_trainval.json'
    img_prefix = data_root
    show_coco(data_root, ann_file, img_prefix)
