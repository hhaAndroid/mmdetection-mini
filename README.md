export PYTHONPATH="$(pwd)"
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29500 tools/train.py configs/retinanet_r50_fpn_1x_coco_iter.py --launcher pytorch
