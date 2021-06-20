import argparse
from collections import OrderedDict

import torch


def convert(src,dst):
    data = torch.load(src)['state_dict']

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

    data = {"state_dict": new_state_dict}
    torch.save(data, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('model', help='model name')
    parser.add_argument('src', help='src yolov5 model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()

