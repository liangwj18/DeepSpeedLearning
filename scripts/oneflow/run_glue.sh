export TASK_NAME=mrpc


# 单机单卡
export CUDA_VISIBLE_DEVICES=0
#python oneflow_proj/test_glue.py \
#    --model_name_or_path bert-base-cased \
#    --task_name $TASK_NAME \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --learning_rate 2e-5 \
#    --num_train_epochs 3 \
#    --overwrite_output_dir yes \
#    --output_dir ./output/python_three/$TASK_NAME/

# 数据并行训练：单机多卡
#export CUDA_VISIBLE_DEVICES=0,6
#python3 -m oneflow.distributed.launch --nproc_per_node 2 oneflow_proj/test_glue.py \
#    --model_name_or_path bert-base-cased \
#    --task_name $TASK_NAME \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --learning_rate 2e-5 \
#    --num_train_epochs 3 \
#    --overwrite_output_dir yes \
#    --output_dir ./output/python_three/$TASK_NAME/


# 数据并行训练：多机多卡
export CUDA_VISIBLE_DEVICES=0
python3 -m oneflow.distributed.launch --nnodes 2 --node_rank 0 --nproc_per_node 1 \
    --master_addr 10.103.10.158 --master_port 22222 oneflow_proj/test_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --overwrite_output_dir yes \
    --output_dir ./output/python_three/$TASK_NAME/


# python run_glue.py \
#   --model_name_or_path bert-base-cased \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir /tmp/$TASK_NAME/