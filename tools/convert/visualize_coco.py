# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def showBBox(coco, anns, label_box=True):
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
        # option for dash-line
        # ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=c, linewidth=2))
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
    # option for filling bounding box
    # p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    # ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_coco(data_dir, datasets, only_bbox, show_all, category_name='bicycle'):
    annotation_file = os.path.join(data_dir, datasets)
    example_coco = COCO(annotation_file)

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
        path = os.path.join(DATA_DIR, '', image_data['file_name'])
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
    DATA_DIR = "/home/pi/dataset/VOCdevkit/"
    DATASETS = "annotations/voc0712_trainval.json"
    ONLY_BBOX = True
    SHOW_ALL = True
    show_coco(DATA_DIR, DATASETS, ONLY_BBOX, SHOW_ALL)
