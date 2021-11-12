import cv2
from ..builder import PIPELINES
import numpy as np

__all__ = ['Yolov5Resize', 'LetterResize']


@PIPELINES.register_module()
class Yolov5Resize:
    def __init__(self,
                 img_scale=None,
                 backend='cv2'):
        self.backend = backend
        self.img_scale = img_scale

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                                 interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
            results[key] = img
        return results


# from https://github.com/ultralytics/yolov5

# 第一步： 计算缩放比例，假设input_shape = (181, 110, 3)，输出shape=201，先计算缩放比例1.11和1.9,选择小比例
#         这个是常规操作，保证缩放后最长边不超过设定值
# 第二步： 计算pad像素，前面resize后会变成(201,122,3)，理论上应该pad=(0,79)，采用最小pad原则，设置最多不能pad超过64像素
#         故对79采用取模操作，变成79%64=15，然后对15进行/2，然后左右pad即可
# 原因是：在单张推理时候不想用letterbox的正方形模式，而是矩形模式，可以加快推理时间、但是在batch测试中，会右下pad到整个batch内部wh最大值
@PIPELINES.register_module()
class LetterResize(object):
    def __init__(self,
                 img_scale=None,
                 color=(114, 114, 114),
                 auto=True,
                 scaleFill=False,
                 scaleup=True,
                 backend='cv2'):
        self.image_size_hw = img_scale
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.backend = backend

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]

            if 'batch_shape' in results:
                batch_shape = results['results']
                self.image_size_hw = batch_shape[::-1]

            shape = img.shape[:2]  # current shape [height, width]
            if isinstance(self.image_size_hw, int):
                self.image_size_hw = (self.image_size_hw, self.image_size_hw)

            # Scale ratio (new / old)
            r = min(self.image_size_hw[0] / shape[0], self.image_size_hw[1] / shape[1])
            if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
                r = min(r, 1.0)
            ratio = r, r
            # 保存图片宽高缩放的最佳size
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            # 为了得到指定输出size，可能需要pad,pad参数
            dw, dh = self.image_size_hw[1] - new_unpad[0], self.image_size_hw[0] - new_unpad[1]  # wh padding
            if self.auto:  # minimum rectangle
                dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
            elif self.scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                # 直接强制拉伸成指定输出
                new_unpad = (self.image_size_hw[1], self.image_size_hw[0])
                ratio = self.image_size_hw[1] / shape[1], self.image_size_hw[0] / shape[0]  # width, height ratios

            # 左右padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2

            # 没有padding前
            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            results['img_shape'] = img.shape
            scale_factor = np.array([ratio[0], ratio[1], ratio[0], ratio[1]], dtype=np.float32)
            results['scale_factor'] = scale_factor

            # padding
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border

            results[key] = img

            results['pad_shape'] = img.shape
            results['pad_param'] = np.array([top, bottom, left, right], dtype=np.float32)
        return results
