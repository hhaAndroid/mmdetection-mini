# Model Zoo
## retinanet
和mmdetection完全一致
## yolov3
和mmdetection完全一致
## darknet中yolo系列
目前支持yolov3/v4和tiny-yolov3/v4的模型  
对比结果如下：

测试配置为：
```python 
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_thr=0.45),
        max_per_img=100)
```

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

附加内容：  
mmdetection: 416x416 考虑scale_x_y = 1.05 37.9% mAP@0.5 (19.3 mAP@0.5:0.95) -24.3 MB  

修改配置为：

```python 
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.000000005,
        conf_thr=0.001,
        nms=dict(type='nms', iou_thr=0.45),
        max_per_img=100)
```
参考： https://github.com/AlexeyAB/darknet/tree/master/src/yolo_layer.c  
mmdetection: 416x416 考虑scale_x_y = 1.05 38.0% mAP@0.5 (19.3 mAP@0.5:0.95) -24.3 MB 


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


附加内容：   
mmdetection: 416x416 考虑scale 65.8% mAP@0.5 (42.0% AP@0.5:0.95) -257. MB   
mmdetection: 608x608 考虑scale 73.0% mAP@0.5 (48.4% AP@0.5:0.95) -257.7 MB     


### 附加说明

- darknet权重转mmdetection的脚本在tools/darknet文件夹里面
- 模型实现参考了https://github.com/Tencent/ObjectDetection-OneStageDet


## yolov5
目前采用的是yolov5 v3发行版本

yolov5采用了pytorch1.6的版本，其包括nn.Hardswish()函数，而pytorch1.3没有
故我采用了替代版本，效果测试后是一样的。

640x640，采用coco2017的val2017测试

yolov5参数： conf_thres=0.001 iou_thres=0.65  
mmdetection: 

```python
   test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.001,
        nms=dict(type='nms', iou_thr=0.65),
        max_per_img=100)  
```



orig yolov5s: 37.0@mAP0.5...0.9 56.2@mAP0.  
mmdetection: 36.0@mAP0.5...0.9 56.3@mAP0.5  



可以看出有一点差距，原因可能有：图片前逻辑处理不一样，yolov5是采用letterbox方式(pad指定值)，
而mmdetection是直接保持长宽比进行resize，没有pad操作。  

后续有空我会把letterbox操作写到mmdetection里面，就可以完全相同了  

我将mmdetection-mini的骨架加上head部分代码嵌入到yolov5中，切换不同模型测试
，结果和原始yolov5完全相同，说明整个模型部分代码没有任何问题。如下图所示：


后来发现mmdetection还有一个score_thr阈值，而yolov5是没有这个参数的
故将该参数设置的极小进行测试score_thr=0.0000001

```python
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.0000001,
        conf_thr=0.001,
        nms=dict(type='nms', iou_thr=0.6),
        max_per_img=300)
```

结果如下：  

orig yolov5s: 37.0@mAP0.5...0.9 56.2@mAP0.    
mmdetection: 36.6@mAP0.5...0.9 56.6@mAP0.5   

指标就非常接近了  

当切换为letterresize模式，采用同样配置  

orig yolov5s: 37.0@mAP0.5...0.9 56.2@mAP0.    
mmdetection: 36.6@mAP0.5...0.9 56.5@mAP0.5   

看来不是letterresize问题，差别几乎没有。剩下差距的0.4个点就不清楚了。 



其余模型：  测试参数为letterresize模式，且配置为：  
```python
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.0000001,
        conf_thr=0.001,
        nms=dict(type='nms', iou_thr=0.6),
        max_per_img=300)
```


orig yolov5m: 44.3@mAP0.5...0.9 63.2@mAP0.5  
mmdetection: 43.9@mAP0.5...0.9 63.4@mAP0.5  



orig yolov5l: 47.7@mAP0.5...0.9 66.5@mAP0.5  
mmdetection: 47.1@mAP0.5...0.9 66.2@mAP0.5  



orig yolov5x: 49.2@mAP0.5...0.9 67.7@mAP0.5  
mmdetection: 48.6@mAP0.5...0.9 67.6@mAP0.5  


再次仔细检查，发现letterresize虽然是用了，但是其输入shape是自适应的，其保证了训练和测试的数据处理逻辑一样(除了mosaic逻辑外)也就是说yolov5的
测试模式下，每个batch内部的shape是不一样的。这个才是造成最终差异的原因。

而在detertor代码里面，是直接调用letterresize，而输入shape是指定的，所以才会找出
我们在对某一张图进行demo 测试时候，结果完全相同，但是test代码时候mAP不一致。



yolov5采用dataloader进行测试时候，实际上是有自适应的，虽然你设置的是640的输入，其流程是：

1. 遍历所有验证集图片的shape，保存起来
2. 开启Rectangular模式，对所有shape按照h/w比例从小到大排序
3. 计算所有验证集，一共可以构成多少个batch，然后对前面排序后的shape进行连续截取操作，并且考虑h/w大于1和小于1的场景，
因为h/w不同，pad的方向也不同，保存每个batch内部的shape比例都差不多
4. 将每个batch内部的shape值转化为指定的图片大小比例，例如打算网络预测最大不超过640，那么所有shape都要不大于640
5. 对batch内部图片进行letterbox操作，测试或者训练时候，不开启minimum rectangle操作。也就是输出shape一定等于指定的shape。这样可以保证
每个batch内部输出的图片shape完全相同


而mmdetection中test时候实现的逻辑是：
1. 将每张图片LetterResize到640x640(输出不一定是640x640)
2. 将图片shape pad到32的整数倍，右下pad
3. 在collate函数中将一个batch内部的图片全部右下pad到当前batch最大的w和h，变成相同shape

可以看出yolov5这种设置会更好一点。应该就是这个差异导致的mAP不一样。


关于yolov5中心点解码问题的分析：  
yolov5中是：  
x_center_pred = (pred_bboxes[..., 0] * 2. - 0.5 + grid[:, 0]) * stride  # xy    
y_center_pred = (pred_bboxes[..., 1] * 2. - 0.5 + grid[:, 1]) * stride  # xy    

pred_bboxes的xy是相对网格左上角的偏移，而mmdetection中anchor是相对网格中心，以x方向为例  
 (x*2-0.5+grid_x)*s  
= (x*2-0.5)*s+grid_x*s   # yolov5  
= (x*2-0.5)*s+(xcenter/s-0.5)*s  
= (x-0.5)*2*s+xcenter  # mmdetection  


关于yolov4的scale_xy，中心点解码问题的分析：  
yolov4中是：  
x_center_pred = (pred_bboxes[..., 0] * self.scale_x_y - 0.5 * (self.scale_x_y - 1)  + grid[:, 0]) * stride  # xy       

pred_bboxes的xy是相对网格左上角的偏移，而mmdetection中anchor是相对网格中心，以x方向为例     

 (x*1.05-0.5(1.05-1)+grid)*s   
= (x*1.05-0.5(1.05-1))*s+grid_x*s   
= (x*1.05-0.5(1.05-1))*s+(xcenter/s-0.5)*s     
= (x*1.05-0.5(1.05-1)-0.5)*s+xcenter   