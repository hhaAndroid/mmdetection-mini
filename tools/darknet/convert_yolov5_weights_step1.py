import torch


# 该程序必须用pytorch1.6的版本才能读取

if __name__ == '__main__':
    weights_file = '/home/pi/pytorch_2020/yolov5/yolov5s.pt'
    weight = torch.load(weights_file, map_location='cpu')['model']  # 只能用pytorch1.6才能加载
    print('cfg', weight.yaml)
    print('stride', weight.stride)
    print('hyp', weight.hyp)
    print('model', weight.model)
    print('num_class', weight.nc)

    model = weight.model
    state_dict = model.state_dict()
    data = {"state_dict": state_dict}
    print(state_dict.keys())
    # pytorch1.6的保存格式更改了，低版本无法读取，需要采用_use_new_zipfile_serialization
    torch.save(data, '../../yolov5s.pth', _use_new_zipfile_serialization=False)

