import datetime
import json
import os
from .pycococreatortools import create_image_info, create_annotation_info


class CocoCreator(object):
    def __init__(self, categories, year=2020, out_dir='./', save_name='temp.json'):
        INFO = {
            "description": "Dataset",
            "url": "https://github.com/hhaAndroid/mmdetection-mini",
            "version": "1.0.0",
            "year": year,
            "contributor": "hha",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }
        LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]
        self.check_categories(categories)
        self.coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": categories,
            "images": [],
            "annotations": []
        }

        self.out_dir = out_dir
        self.save_name = save_name

    def check_categories(self, categories):
        """
        example:
        [
        {
            'id': 1, # 类别1
            'name': 'power',
            'supercategory': 'object',
        }
        {
            'id': 2, # 类别2
            'name': 'circle',
            'supercategory': 'shape',
        }
        ]

        """
        assert isinstance(categories, list)
        assert isinstance(categories[0], dict)

    def create_image_info(self, image_id, file_name, image_size):
        image_info = create_image_info(image_id, file_name, image_size)
        self.coco_output["images"].append(image_info)

    def create_annotation_info(self, segmentation_id, image_id, category_info, binary_mask=None, bounding_box=None,
                               image_size=None, tolerance=2):
        annotation_info = create_annotation_info(segmentation_id, image_id, category_info, binary_mask, image_size,
                                                 tolerance,
                                                 bounding_box)
        if annotation_info is not None:
            self.coco_output["annotations"].append(annotation_info)

    def dump(self):
        out_file = os.path.join(self.out_dir, 'annotations', self.save_name)
        with open(out_file, 'w') as output_json_file:
            json.dump(self.coco_output, output_json_file)
