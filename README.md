export PYTHONPATH="$(pwd)"

SRUN_ARGS='--quotatype=spot' GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=4  ./tools/slurm_train.sh mm_det job configs/retinanet/retinanet_r50_caffe_fpn_mstrain_1x_iter_coco.py ./tools/work_dir/retinanet_r50_caffe_fpn_mstrain_1x_iter_coco

GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2  ./tools/slurm_train.sh mediaf1 job5 configs/yolov5/yolov5s_v6.py ./tools/work_dir/yolov5/yolov5s_v6.py
