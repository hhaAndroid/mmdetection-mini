export PYTHONPATH="$(pwd)"

SRUN_ARGS='--quotatype=spot' GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=4  ./tools/slurm_train.sh mm_det job configs/retinanet/retinanet_r50_caffe_fpn_mstrain_1x_iter_coco.py ./tools/work_dir/retinanet_r50_caffe_fpn_mstrain_1x_iter_coco
