import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 控制图片是否进行上采样，默认是 True
    # 如果设置为 False, 表示只能进行下采样，不能进行上采样
    # 如果输入图片小于 new_shape 值，那么啥也不干，后续只会进行 pad，而不会 resize
    # 如果输入图片大于 new_shape 值，则会进行下采样
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        # 如果为 True, 那么会保证能够被 stride 整除，输出是最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        # 强制拉伸到 new_shape，不进行任何 padding，默认是 False
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # 是否要 resize
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def demo_1(img):
    h0, w0 = img.shape[:2]  # orig hw
    r = 640 / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        # 缩小图像用 cv2.INTER_AREA
        # 放大图像用 cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                         interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    print(img.shape)
    img, ratio, pad = letterbox(img, (640, 640), auto=False, scaleup=False)
    print(img.shape)


if __name__ == '__main__':
    # mm_letterbox = LetterResize()

    # 0 scaleFill 参数可以忽略，一般不用，不用管
    # 1 auto=True and scaleup=True，表示图片可以进行上采样，并且输出 shape 要保证能够被 stride 参数整除，且为最小矩形，输出 shape 不一定和 new_shape 一样
    # 2 auto=False and scaleup=True，表示图片可以进行上采样。如果有必要则先进行保持宽高比例的上下采样，然后 padding 到指定 new_shape 输出
    # 3 auto=True and scaleup=False，表示图片只能进行下采样。如果输入图片比较小，则只能通过 pad 操作保证被 stride 参数整除，如果输入图片比较大，则会先进行保持比例下采样，然后保证整除
    # 4 auto=False and scaleup=False，表示图片只能进行下采样。如果输入图片比较小，则只能通过 pad 操作使得输出变成 new_shape，如果输入图片比较大，则会先进行保持比例下采样，然后pading变成new_shape
    # 综上所述，常用配置是 1

    image_w = 100
    image_h = 201
    img = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    img_out = letterbox(img, auto=False, scaleup=True)[0]
    # print(img_out.shape)
    demo_1(img)


