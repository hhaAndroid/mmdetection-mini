import numpy as np
import cv2

from mmdet import cv_core

import torch
import torch.nn.functional as F



def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = cv_core.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def merge_imgs(imgs, row_col_num):
    """
        Merges all input images as an image with specified merge format.

        :param imgs : img list
        :param row_col_num : number of rows and columns displayed
        :return img : merges img
        """

    length = len(imgs)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
    color = random_color(rgb=True).astype(np.float64)

    for img in imgs:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

    if row_col_num[1] < 0 or length < row:
        merge_imgs = np.hstack(imgs)
    elif row_col_num[0] < 0 or length < col:
        merge_imgs = np.vstack(imgs)
    else:
        assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

        fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start: end])
            merge_imgs_col.append(merge_col)

        merge_imgs = np.vstack(merge_imgs_col)

    return merge_imgs


def show_img(imgs, window_names=None, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
    """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.

        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
    if not isinstance(imgs, list):
        imgs = [imgs]

    if window_names is None:
        window_names = list(range(len(imgs)))
    else:
        if not isinstance(window_names, list):
            window_names = [window_names]
        assert len(imgs) == len(window_names), 'window names does not match images!'

    if is_merge:
        merge_imgs1 = merge_imgs(imgs, row_col_num)

        cv2.namedWindow('merge', 0)
        cv2.imshow('merge', merge_imgs1)
    else:
        for img, win_name in zip(imgs, window_names):
            if img is None:
                continue
            win_name = str(win_name)
            cv2.namedWindow(win_name, 0)
            cv2.imshow(win_name, img)

    cv2.waitKey(wait_time_ms)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def show_bbox(image, bboxs_list, color=None,
              thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.

    :param image: numpy.ndarray.
    :param bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
    :param color: tuple.
    :param thickness: int.
    :param fontScale: float.
    :param wait_time_ms: int
    :param names: string: window name
    :param is_show: bool: whether to display during middle process
    :return: numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.copy()
    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            colors = random_color(rgb=True).astype(np.float64)
        else:
            colors = color

        if not is_without_mask:
            image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                       thickness)
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                        font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    if is_show:
        show_img(image_copy, names, wait_time_ms)
    return image_copy


def show_tensor(tensor, resize_hw=None, top_k=50, mode='CHW', is_show=True,
                wait_time_ms=0, show_split=True, is_merge=True, row_col_num=(1, -1)):
    """

        :param wait_time_ms:
        :param tensor: torch.tensor
        :param resize_hw: list:
        :param top_k: int
        :param mode: string: 'CHW' , 'HWC'
        """

    def normalize_numpy(array):
        max_value = np.max(array)
        min_value = np.min(array)
        array = (array - min_value) / (max_value - min_value)
        return array

    assert tensor.dim() == 3, 'Dim of input tensor should be 3, please check your tensor dimension!'

    # 默认tensor格式,通道在前
    if mode == 'CHW':
        tensor = tensor
    else:
        tensor = tensor.permute(2, 0, 1)

    # 利用torch中的resize函数进行插值, 选择双线性插值平滑
    if resize_hw is not None:
        tensor = tensor[None]
        tensor = F.interpolate(tensor, resize_hw, mode='bilinear')
        tensor = tensor.squeeze(0)

    tensor = tensor.permute(1, 2, 0)

    channel = tensor.shape[2]

    if tensor.device == 'cpu':
        tensor = tensor.detach().numpy()
    else:
        tensor = tensor.cpu().detach().numpy()
    if not show_split:
        # sum可能会越界，所以需要归一化
        sum_tensor = np.sum(tensor, axis=2)
        sum_tensor = normalize_numpy(sum_tensor) * 255
        sum_tensor = sum_tensor.astype(np.uint8)

        # 热力图显示
        sum_tensor = cv2.applyColorMap(np.uint8(sum_tensor), cv2.COLORMAP_JET)
        # mean_tensor = cv2.applyColorMap(np.uint8(mean_tensor), cv2.COLORMAP_JET)

        if is_show:
            show_img([sum_tensor], ['sum'], wait_time_ms=wait_time_ms)
        return [sum_tensor]
    else:
        assert top_k > 0, 'top k should be positive!'
        channel_sum = np.sum(tensor, axis=(0, 1))
        index = np.argsort(channel_sum)
        select_index = index[:top_k]
        tensor = tensor[:, :, select_index]
        tensor = np.clip(tensor, 0, np.max(tensor))

        single_tensor_list = []
        if top_k > channel:
            top_k = channel
        for c in range(top_k):
            single_tensor = tensor[..., c]
            single_tensor = normalize_numpy(single_tensor) * 255
            single_tensor = single_tensor.astype(np.uint8)

            single_tensor = cv2.applyColorMap(np.uint8(single_tensor), cv2.COLORMAP_JET)
            single_tensor_list.append(single_tensor)

        if is_merge:
            return_imgs = merge_imgs(single_tensor_list, row_col_num=row_col_num)
        else:
            return_imgs = single_tensor_list

        if is_show:
            show_img(return_imgs, wait_time_ms=wait_time_ms, is_merge=is_merge)
        return return_imgs
