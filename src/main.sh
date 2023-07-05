export WANDB_PROJECT="attacker-eval"
export CUDA_VISIBLE_DEVICES="5,6"
export TASK=multi_nli
export TASK_NAME=mrpc
export TOKENIZERS_PARALLELISM=true

python main.py \
  --seed 112 \
  --model_name_or_path facebook/bart-large \
  --target_model_name_or_path facebook/bart-large-mnli \
  --task $TASK \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ../model/$TASK