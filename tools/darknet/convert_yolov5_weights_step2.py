import torch
from collections import OrderedDict

# 该程序必须用mmdetection环境读取

if __name__ == '__main__':
    weights_file = '../../yolov5s.pth'
    data = torch.load('../../yolov5s.pth')['state_dict']
    # print(data.keys())

    # 不同的模型 10和24需要改

    new_state_dict = OrderedDict()
    for k, v in data.items():
        num = int(k.split('.')[0])
        if num < 10:  # 前9个是骨架
            name = 'backbone.backbone.' + k
        elif num < 24:  # neck
            name = 'bbox_head.det.' + str(num - 10) + k[2:]
        else:  # head
            name = 'bbox_head.head.' + k[5:]

        if k.find('anchors') >= 0 or k.find('anchor_grid') >= 0:
            continue

        new_state_dict[name] = v

    print(new_state_dict.keys())
    data = {"state_dict": new_state_dict}
    torch.save(data, '../../yolo5s_mm.pth')
