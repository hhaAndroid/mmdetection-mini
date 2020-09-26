import torch
import os
# from tools.darknet.yolov5_download import attempt_download
from utils.google_utils import attempt_download


# 注意本程序无法运行！！！
# 该程序必须放在yolov5工程目录下，用pytorch1.6的版本才能读取

if __name__ == '__main__':
    # v3版本下载地址 https://github.com/ultralytics/yolov5/releases/tag/v3.0

    is_download = True
    save_name = 'yolov5s.pth'

    if is_download:
        name = 'yolov5s.pt'
        attempt_download(name)
        weights_file = name
    else:
        weights_file = '/home/pi/yolov5s.pt'

    weight = torch.load(weights_file, map_location=torch.device('cpu'))['model']  # 只能用pytorch1.6才能加载
    print('cfg', weight.yaml)
    print('stride', weight.stride)
    print('hyp', weight.hyp)
    print('model', weight.model)
    print('num_class', weight.nc)

    model = weight.model
    state_dict = model.state_dict()
    data = {"state_dict": state_dict}
    # print(state_dict.keys())
    # pytorch1.6的保存格式更改了，低版本无法读取，需要采用_use_new_zipfile_serialization
    torch.save(data, save_name, _use_new_zipfile_serialization=False)
