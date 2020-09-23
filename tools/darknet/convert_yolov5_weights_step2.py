import torch
from collections import OrderedDict

# 该程序必须用mmdetection环境读取

if __name__ == '__main__':

    name = 'yolov5s'  # yolov5s yolov5m yolov5l yolov5x

    weights_file = '/home/pi/pytorch_2020/yolov5/yolov5s.pth'
    save_name = '../../yolov5s_mm.pth'

    data = torch.load(weights_file)['state_dict']

    index = [10, 24]

    new_state_dict = OrderedDict()
    for k, v in data.items():
        num = int(k.split('.')[0])
        if num < index[0]:  # 前9个是骨架
            name = 'backbone.backbone.' + k
        elif num < index[1]:  # neck
            name = 'bbox_head.det.' + str(num - index[0]) + k[2:]
        else:  # head
            name = 'bbox_head.head.' + k[5:]

        if k.find('anchors') >= 0 or k.find('anchor_grid') >= 0:
            continue

        new_state_dict[name] = v

    # print(new_state_dict.keys())
    data = {"state_dict": new_state_dict}
    torch.save(data, save_name)
