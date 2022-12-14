#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

#wget https://docs.oneflow.org/master/code/parallelism/ddp_train.py #下载脚本

# 数据并行训练：单机双卡
#python3 -m oneflow.distributed.launch oneflow_proj/test_mnist.py --nproc_per_node 2



# 数据并行训练：多机单卡
python3 -m oneflow.distributed.launch --nnodes 2 --node_rank 0 --nproc_per_node 1 \
    --master_addr 10.103.10.158 --master_port 22222 \
    oneflow_proj/test_mnist.py


