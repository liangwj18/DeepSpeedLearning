#!/bin/bash

#python3 -m oneflow.distributed.launch --nproc_per_node 2 ./script.py


#python3 -m oneflow_proj.distributed.launch \
#    --nnodes=2 \
#    --node_rank=0 \
#    --nproc_per_node=2 \
#    --master_addr="192.168.1.1" \
#    --master_port=7788 \
#    script.py


#python3 -m oneflow_proj.distributed.launch \
#    --nnodes=2 \
#    --node_rank=1 \
#    --nproc_per_node=2 \
#    --master_addr="192.168.1.1" \
#    --master_port=7788 \
#    script.py