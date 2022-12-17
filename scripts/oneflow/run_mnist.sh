#!/bin/bash

#export CUDA_VISIBLE_DEVICES=7

#wget https://docs.oneflow.org/master/code/parallelism/ddp_train.py #下载脚本

# 单机单卡


# 数据并行训练：单机双卡
#export CUDA_VISIBLE_DEVICES=0
#python3 -m oneflow.distributed.launch --nproc_per_node 1 oneflow_proj/test_mnist.py



# 数据并行训练：多机单卡
export CUDA_VISIBLE_DEVICES=1,7
python3 -m oneflow.distributed.launch --nnodes 2 --node_rank 0 --nproc_per_node 2 \
    --master_addr 10.103.10.158 --master_port 22222 \
    oneflow_proj/test_mnist.py --bsz 64


