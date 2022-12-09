export CUDA_VISIBLE_DEVICES=3

python3 -m torch.distributed.launch --nproc_per_node=1 \
    --nnodes=2 --node_rank=0 --master_addr="10.103.10.158" \
    --master_port=12346 \
    train.py \
    --deepspeed_config=./ds_config.json \



# deepspeed --hostfile=hostfile \
# train.py \
# --deepspeed_config=./ds_config.json \
# -p 2 \
# --steps=200 \