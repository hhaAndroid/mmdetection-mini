# RepPoints V2: Verification Meets Regression for Object Detection

By Yihong Chen, [Zheng Zhang](https://stupidzz.github.io/), [Yue Cao](http://yue-cao.me/), [Liwei Wang](http://www.liweiwang-pku.com/), [Stephen Lin](https://scholar.google.com/citations?hl=zh-CN&user=c3PYmxUAAAAJ), [Han Hu](https://ancientmooner.github.io/).

We provide supported codes and configuration files to reproduce ["RepPoints V2: Verification Meets Regression for Object Detection"](https://arxiv.org/abs/2007.08508) on COCO object detection and instance segmentation. Besides, this repo also includes improved results for [RepPoints V1](https://arxiv.org/pdf/1904.11490.pdf), [Dense RepPoints](https://arxiv.org/pdf/1912.11473.pdf) (V1,V2). Our code is adapted from [mmdetection](https://github.com/open-mmlab/mmdetection). 

## Introduction

Verification and regression are two general methodologies for prediction in neural networks. Each has its own strengths: verification can be easier to infer accurately, and regression is more efficient and applicable to continuous target variables. Hence, it is often beneficial to carefully combine them to take advantage of their benefits. We introduce verification tasks into the localization prediction of RepPoints, producing **RepPoints v2**. 

RepPoints v2 aims for object detection and it achieves `52.1 bbox mAP` on COCO test-dev by a single model. Dense RepPoints v2 aims for instance segmentation and it achieves `45.9 bbox mAP` and `39.0 mask mAP` on COCO test-dev by using a ResNet-50 model.

<div align="center">
  <img src="demo/reppointsv2.png" width="1178" />
</div>

## Main Results

### RepPoints V2

**ResNe(X)ts:**

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link 
--- |:---:|:---:|:---:|:---:
RepPoints_V2_R_50_FPN_1x | No | 40.9 | --- | [Google](https://drive.google.com/file/d/1QBYTLITOJG5dSjU35YewE9efSCH_VGg2/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1ZvJ3gk_FVOVHmvmy87cr_w) / [Log](https://drive.google.com/file/d/1Ra2XC-Zjfpx6YG91ZRY_8qI_XnDe_Txu/view?usp=sharing)
RepPoints_V2_R_50_FPN_GIoU_1x | No  | 41.1 | 41.3 | [Google](https://drive.google.com/file/d/1lbYUpvA33GHaEImKRhbR7H5S36Dxubcf/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1kyt5YNWO-gg_W4iUuwZxiw) / [Log](https://drive.google.com/file/d/1yDwNToYAZdPWTs4vxzbBNqRCLIrRxx_H/view?usp=sharing)
RepPoints_V2_R_50_FPN_GIoU_2x | Yes  | 43.9 | 44.4 | [Google](https://drive.google.com/file/d/13FfoXOfTsO-eLTcO__WXUxQRRWNzAdrL/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1QAvjzGI1zrnockXX6cZrZA) / [Log](https://drive.google.com/file/d/1yDwNToYAZdPWTs4vxzbBNqRCLIrRxx_H/view?usp=sharing)
RepPoints_V2_R_101_FPN_GIoU_2x | Yes  | 45.8 | 46 | [Google](https://drive.google.com/file/d/1MUb1Y1_OoqhwFkvdyE6QthbUc1l2ixYS/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1YNxbnmeq20mVef5ZAlMTxQ) / [Log](https://drive.google.com/file/d/1m5BM1PWXWKwfsvEc54b0fCcMIlXKPFG_/view?usp=sharing)
RepPoints_V2_R_101_FPN_dcnv2_GIoU_2x | Yes  | 47.7 | 48.1 | [Google](https://drive.google.com/file/d/1VaBAPWOzku0tfpUWa5FmOsu_Fn_3UWaY/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/14V2hz6VrXJv_acQ-SQlBUg) / [Log](https://drive.google.com/file/d/1f_WwyNFlqvCSU-P723b-LHG1kpRhddns/view?usp=sharing)
RepPoints_V2_X_101_FPN_GIoU_2x | Yes  | 47.3 | 47.8 | [Google](https://drive.google.com/file/d/1rThw_7yXi185-VXfeXY81iJNwoAywQVA/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1Vp4vtkSSfAbkI1_--jDQ5g) / [Log](https://drive.google.com/file/d/13Nj__4nvZEEJtwNhvIcKjDwbmu5eQZoo/view?usp=sharing)
RepPoints_V2_X_101_FPN_dcnv2_GIoU_2x | Yes  | 49.3 | 49.4 | [Google](https://drive.google.com/file/d/1db6cK7pEjRgN8QjaGV8OnZYp35nuxd7G/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1idrD8kmgYTP_q5_mSSlpTQ) / [Log](https://drive.google.com/file/d/1DlCYQiWUanPVwyowjFLrgKCYz57EJE2S/view?usp=sharing)

**MobileNets**:

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
RepPoints_V2_MNV2_c128_FPN_2x | Yes | 36.8 | --- | [Google](https://drive.google.com/file/d/1mnoXOyzp6dCYbQx7rKbzjKlc0RzitOpV/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1_sEMkhDjYjhJwMSiNNoYdg) / [Log](https://drive.google.com/file/d/11UQGvOuOykFD0iW3xz1DsqJSwJaw2tmw/view?usp=sharing)
RepPoints_V2_MNV2_FPN_2x | Yes | 39.4 | --- | [Google](https://drive.google.com/file/d/1xk8jGZiRs2iskywf3hB6tINMK_3lDL3u/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1OdiEtxWe5f45GaaprITdrA) / [Log](https://drive.google.com/file/d/1A1ldy4HzPStKjz0Xm96sPnyB-NS2wzMk/view?usp=sharing)

### RepPoints V1

**ResNe(X)ts:**

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
RepPoints_V1_R_50_FPN_1x | No | 38.8 | --- | [Google](https://drive.google.com/file/d/1DMoTicyL5FejCL3042rwZWPojTC2-qRH/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1t4zaVFCH0A35xEDXo3RUMQ) / [Log](https://drive.google.com/file/d/1Oq-3DFnfbJ6F5doJobuhZU9y7BkRC0NH/view?usp=sharing)
RepPoints_V1_R_50_FPN_GIoU_1x | No  | 39.9 | ---| [Google](https://drive.google.com/file/d/1IJp3bBCrRuDDQcxjwoUenBuA-KSzYWkf/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1swvcTxgiUWSRCOSKOJ7Mjg) / [Log](https://drive.google.com/file/d/1bRCfSQPYFjxXIari9F8AWXOYUUS4VVJg/view?usp=sharing)
RepPoints_V1_R_50_FPN_GIoU_2x | Yes  | 42.7 | --- | [Google](https://drive.google.com/file/d/1tZpfOGmxzToaikaFdpyhjCjm8ltOJaYY/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1pFvleWnZjVsFeHJjiX3dpw) / [Log](https://drive.google.com/file/d/1iBSrXp4ngug9jaWb1gndSj4NQQQzbpAC/view?usp=sharing)
RepPoints_V1_R_101_FPN_GIoU_2x | Yes  | 44.4 | --- | [Google](https://drive.google.com/file/d/1YiR4m8GNWQ472tgGXOdeaWbnS2KiAR4z/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1BhVjPvBJaWM3Okq1Ytk8Iw) / [Log](https://drive.google.com/file/d/1QhN3vedurGiRAl6aiSwtGNwxN3TAOXuP/view?usp=sharing)
RepPoints_V1_R_101_FPN_dcnv2_GIoU_2x | Yes  | 46.6 | --- | [Google](https://drive.google.com/file/d/112jG1a2TUnABqCR1ccKrIzAE6_8P3LAe/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/18e0zGQah6aqCY2qIXe88Ew) / [Log](https://drive.google.com/file/d/1ut23n60vRY0f8VF97bgh2KWN05rmto9N/view?usp=sharing)
RepPoints_V1_X_101_FPN_GIoU_2x | Yes  | 46.3 | --- | [Google](https://drive.google.com/file/d/1UohtogF-znE0NnqXHSrcGBuE9B3pNaDr/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1qbCyHBZksS_l-eXH8Wh5Ug) / [Log](https://drive.google.com/file/d/1wOBP8-oBg53llJOfspddecA5NNpdcRix/view?usp=sharing)
RepPoints_V1_X_101_FPN_dcnv2_GIoU_2x | Yes  | 48.3 | --- | [Google](https://drive.google.com/file/d/14oSaFilmT6EMTkAuP23OyQtLbWRB0BiN/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1Xl5VUG7z7IQlz6F8267f6A) / [Log](https://drive.google.com/file/d/14u0m5eFdLFRGJT--mx7wzsdIVNb8Atu8/view?usp=sharing)

**MobileNets**:

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
RepPoints_V1_MNV2_c128_FPN_2x | Yes | 35.7 | --- | [Google](https://drive.google.com/file/d/14l_m3lLacfw7mTafv7cuvczhkWeF2cn4/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1hHeV8KZLdtLNvVYRl-najw) / [Log](https://drive.google.com/file/d/1pg1gW3ajOqgB4oLmEsiKJ2cReAIdxRkp/view?usp=sharing)
RepPoints_V1_MNV2_FPN_2x | Yes | 37.8 | --- | [Google](https://drive.google.com/file/d/1Ex6us97waqWP25H20xBZZEfQXfI8mfjW/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1AyibFbRgSc8bug0KZDO1Yw) / [Log](https://drive.google.com/file/d/1BanM614yhFaxtLeSI_vYbtTjsarj-Xxl/view?usp=sharing)

### Dense Reppoints V2

Model | MS training | bbox AP (minival/test-dev) | mask AP (minival/test-dev) | Link
--- |:---:|:---:|:---:|:---:
Dense_RepPoints_V2_R_50_FPN_1x | No | 40.5/--- | 34.8/--- | [Google](https://drive.google.com/file/d/14l_m3lLacfw7mTafv7cuvczhkWeF2cn4/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1W2FyECYBVD5WUvYFIT7stw) / [Log]()
Dense_RepPoints_V2_R_50_FPN_GIoU_1x | No  | 41.5/41.6 | 35.1/35.4| [Google](https://drive.google.com/file/d/14l_m3lLacfw7mTafv7cuvczhkWeF2cn4/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1TlTkJZNwQl_8keFjCnEVEA) / [Log]()
Dense_RepPoints_V2_R_50_FPN_GIoU_3x | Yes  | 45.2/45.9 | 38.3/39.0 | [Google](https://drive.google.com/file/d/17VWteFLHYWJCuGwafCiwlTc0Aga-TgtJ/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1VCAJLin53228fSQM9UjHXw) / [Log]()

### Dense Reppoints V1

Model | MS training | bbox AP (minival/test-dev) | mask AP (minival/test-dev) | Link
--- |:---:|:---:|:---:|:---:
Dense_RepPoints_V1_R_50_FPN_1x | No | 39.9/--- | 33.7/--- | [Google](https://drive.google.com/file/d/1DKcxMSXnpNwQe0cFz0e9C00KdMSTMJR0/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/12rS_fEONPFSgHZvXyKvFIQ) / [Log](https://drive.google.com/file/d/1iD9BX5x0dmzbwWDw_S48Vp-DpMZ9JJNz/view?usp=sharing)
Dense_RepPoints_V1_R_50_FPN_GIoU_1x | No  | 40.9/41.1 | 34.2/34.5| [Google](https://drive.google.com/file/d/1GH3Ut9fzPBwRCOQD6j_uLFVIIs-i09bT/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1TWsRzCTZoC16BwTZbqWJ7A) / [Log](https://drive.google.com/file/d/1xKTY_XmontMaYt8ShhvdbNOoNebzPXH3/view?usp=sharing)
Dense_RepPoints_V1_R_50_FPN_GIoU_3x | Yes  | 44.5/45.0 | 37.4/38.0 | [Google](https://drive.google.com/file/d/11t8P5lGRdpLdZS5UY8kYgDLUTPQXBzlF/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1zf5rdh_5Kz1SrjbtBbjlzQ) / [Log](https://drive.google.com/file/d/114ESFYzhALGj4bL1OylnRvQzEd4jYTVa/view?usp=sharing)

[1] *GIoU means using GIoU loss instead of smooth-l1 loss for the regression branch, which we find improves the final performance.* \
[2] *X-101 denotes ResNeXt-101-64x4d.* \
[3] *1x, 2x, 3x mean the model is trained for 12, 24 and 36 epochs, respectively.* \
[4] *For multi-scale training, the shorter side of images is randomly chosen from [480, 960].* \
[5] *`dcnv2` denotes deformable convolutional networks v2.* \
[6] *c128 denotes the model has 128 (instead of 256) channels in towers.*\
[7] *We use syncbn, GIoU loss for the regression branch to train mobilenet v2 models by default.*


## Citation

```
@article{chen2020reppointsv2,
  title={RepPoints V2: Verification Meets Regression for Object Detection},
  author={Chen, Yihong and Zhang, Zheng and Cao, Yue and Wang, Liwei and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2007.08508},
  year={2020}
}

@inproceedings{yang2019dense,
  title={Dense reppoints: Representing visual objects with dense point sets},
  author={Yang, Ze and Xu, Yinghao and Xue, Han and Zhang, Zheng and Urtasun, Raquel and Wang, Liwei and Lin, Stephen and Hu, Han},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{yang2019reppoints,
  title={RepPoints: Point Set Representation for Object Detection},
  author={Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019}
}
```

## Installation

Please refer to [INSTALL.md](docs/install.md) for installation and dataset preparation.

## Get Started

Please see [GETTING_STARTED.md](docs/getting_started.md) for the basic usage of MMDetection.

### Inference

```shell
./tools/dist_test.sh configs/reppoints_v2/reppoints_v2_r50_fpn_giou_1x_coco.py work_dirs/reppoints_v2_r50_fpn_giou_1x_coco/epoch_12.pth 8 --eval bbox
```

Please note that:
1. If you are using other model, please change the config file and pretrained weights accordingly.

### Train

```shell
./tools/dist_train.sh configs/reppoints_v2/reppoints_v2_r50_fpn_giou_1x_coco.py 8
```

## Contributing to the project

Any pull requests or issues are welcome.
