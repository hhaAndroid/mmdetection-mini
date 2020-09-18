#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import os.path as osp
import json
import argparse
import xml.etree.ElementTree as ET
from mmdet import cv_core


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if 0 < length != len(vars):
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def _convert(xml_list, xml_dir, json_file):
    if isinstance(xml_list, list):
        list_fps = []
        for xml in xml_list:
            list_fps.append(open(xml, 'r'))
    else:
        list_fps = [open(xml_list, 'r')]
        xml_dir = [xml_dir]

    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for i, lines in enumerate(list_fps):
        for line in lines:
            line = line.strip()
            print("Processing %s" % (line + '.xml'))
            xml_f = os.path.join(xml_dir[i], line + '.xml')
            flag_name = xml_dir[i].split('/')[-2] + '/JPEGImages'
            tree = ET.parse(xml_f)
            root = tree.getroot()
            path = get(root, 'path')
            if len(path) == 1:
                filename = os.path.basename(path[0].text)
            elif len(path) == 0:
                filename = get_and_check(root, 'filename', 1).text
            else:
                raise NotImplementedError('%d paths found in %s' % (len(path), line))

            image_id = get_filename_as_int(filename)
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            image = {'file_name': os.path.join(flag_name, filename), 'height': height, 'width': width,
                     'id': image_id}
            json_dict['images'].append(image)
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
                ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
                xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    for lines in list_fps:
        lines.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to coco format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')  # voc根路径 里面存放的是VOC2007和VOC2012两个子文件夹
    parser.add_argument('-o', '--out-dir', help='output path')  # annotations 保存文件夹
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    cv_core.mkdir_or_exist(out_dir)

    year = None
    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        year = '2007'
        years.append(year)
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        year = '2012'
        years.append(year)
    if '2007' in years and '2012' in years:
        year = ['2007', '2012']

    if year == '2007':
        prefix = 'voc07'
        split = ['trainval', 'test']  # train集和test集
    elif year == '2012':
        prefix = 'voc12'
        split = ['train', 'val']  # train集和test集
    elif year == ['2007', '2012']:
        prefix = 'voc0712'
        split = [['trainval', 'train'], ['test', 'val']]  # train集和test集
    else:
        raise NotImplementedError

    for split_ in split:
        if isinstance(split_, list):
            dataset_name = prefix + '_' + split_[0]
        else:
            dataset_name = prefix + '_' + split_
        print('processing {} ...'.format(dataset_name))
        annotations_path = osp.join(out_dir, 'annotations')
        cv_core.mkdir_or_exist(annotations_path)
        out_file = osp.join(annotations_path, dataset_name + '.json')
        if isinstance(split_, list):
            filelists = []
            xml_dirs = []
            for i, s in enumerate(split_):
                filelist = osp.join(devkit_path,
                                    'VOC{}/ImageSets/Main/{}.txt'.format(year[i], s))
                xml_dir = osp.join(devkit_path, 'VOC{}/Annotations'.format(year[i]))
                filelists.append(filelist)
                xml_dirs.append(xml_dir)
        else:
            filelists = osp.join(devkit_path, 'VOC{}/ImageSets/Main/{}.txt'.format(year, split_))
            xml_dirs = osp.join(devkit_path, 'VOC{}/Annotations'.format(year))
        _convert(filelists, xml_dirs, out_file)

    print('Done!')


if __name__ == '__main__':
    main()
