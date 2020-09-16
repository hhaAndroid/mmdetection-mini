# Model Zoo
## retinanet
和mmdetection完全一致
## yolov3
和mmdetection完全一致
## darknet中yolo系列
目前支持yolov3/v4和tiny-yolov3/v4的模型
对比结果如下：

### tiny-yolov3 

权重下载链接： https://github.com/AlexeyAB/darknet 
对应配置： https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg
darknet: 33.1% mAP@0.5 - 345(R) FPS - 5.6 BFlops - 33.7 MB
mmdetection: 36.5% mAP@0.5 -35.4 MB

注意： yolov3-tiny.cfg中最后一个[yolo]节点，mask 应该是 1,2,3，而不是github里面的0,1,2

### tiny-yolov4

权重下载链接： https://github.com/AlexeyAB/darknet 
对应配置： https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
darknet: 40.2% mAP@0.5 - 371(1080Ti) FPS / 330(RTX2070) FPS - 6.9 BFlops - 23.1 MB
mmdetection: 37.9% mAP@0.5 -24.3 MB


### yolov3

权重下载链接： https://github.com/AlexeyAB/darknet 
对应配置： https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
darknet:  55.3% mAP@0.5 - 66(R) FPS - 65.9 BFlops - 236 MB
mmdetection: 66.8 mAP@0.5 -248 MB


### yolov4

权重下载链接： https://github.com/AlexeyAB/darknet 
对应配置： https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
darknet:  62.8% mAP@0.5 (41.2% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS - 60.1 BFlops  245 MB
mmdetection: 65.8% mAP@0.5 (20.3% AP@0.5:0.95) -257. MB

注意： yolov4的anchor尺寸变了，不同于yolov3

### 附加说明

- 以上全部是416x416单尺度测试
- darknet权重是在coco的trainval数据上面训练，在test_dev上面测试的结果
而我没有下载test_dev数据，而是采用了val数据进行测试的，所有两者指标没有非常强的对比性。
只能说明我重写的yolo系列代码没有问题，而且可以直接导入coco权重进行微调训练。
- darknet权重转mmdetection的脚本在tools/darknet文件夹里面


