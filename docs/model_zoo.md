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
darknet: 416x416 40.2% mAP@0.5 - 371(1080Ti) FPS / 330(RTX2070) FPS - 6.9 BFlops - 23.1 MB  
mmdetection: 416x416 37.9% mAP@0.5 (19.2 mAP@0.5:0.95) -24.3 MB

注意：更低的原因应该是该配置里面有一个scale_x_y = 1.05一些参数不一样，目前没有利用到


### yolov3

权重下载链接： https://github.com/AlexeyAB/darknet   
对应配置： https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg  
darknet(test_dev): 416x416 55.3% mAP@0.5 (31.0 mAP@0.5:0.95)  - 66(R) FPS - 65.9 BFlops - 236 MB  
darknet(val2017): 416x416 65.9 mAP@0.5
mmdetection(val2017): 416x416 66.8 mAP@0.5 (37.4 mAP@0.5:0.95)  -248 MB


### yolov4

权重下载链接： https://github.com/AlexeyAB/darknet   
对应配置： https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg  

darknet:  416x416 62.8% mAP@0.5 (41.2% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS - 60.1 BFlops  245 MB  
darknet:  608x608 65.7% mAP@0.5 (43.5% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS - 60.1 BFlops  245 MB 
 
mmdetection: 416x416 65.7% mAP@0.5 (41.7% AP@0.5:0.95) -257. MB  
mmdetection: 608x608 72.9% mAP@0.5 (48.1% AP@0.5:0.95) -257.7 MB  

注意： yolov4的anchor尺寸变了，不同于yolov3， 下载的权重是608x608训练过的,测试用了两种尺度而已


### 附加说明

- darknet权重转mmdetection的脚本在tools/darknet文件夹里面
- 模型实现参考了https://github.com/Tencent/ObjectDetection-OneStageDet


