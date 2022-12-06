export TASK_NAME=mrpc
export CUDA_VISIBLE_DEVICES=2,3,6

python run_glue.py \
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
  #   --deepspeed ./ds_config_zero3.json \

#   deepspeed examples/pytorch/translation/run_translation.py \
# --deepspeed tests/deepspeed/ds_config_zero3.json \
# --model_name_or_path t5-small --per_device_train_batch_size 1 \
# --output_dir output_dir --overwrite_output_dir --fp16 \
# --do_train --max_train_samples 500 --num_train_epochs 1 \
# --dataset_name wmt16 --dataset_config "ro-en" \
# --source_lang en --target_lang ro


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
