export PYTHONPATH="$(pwd)"
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29500 tools/train.py configs/retinanet_r50_fpn_1x_coco_iter.py --launcher pytorch

export PYTHONPATH="$(pwd)"
srun -p mm_det --job-name=test --gres=gpu:2 --ntasks=2 --ntasks-per-node=8 --cpus-per-task=2 --kill-on-bad-exit=1 python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco_iter.py