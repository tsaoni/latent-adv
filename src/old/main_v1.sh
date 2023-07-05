export WANDB_PROJECT="latent-adv"
export CUDA_VISIBLE_DEVICES="5"
export TOKENIZERS_PARALLELISM=false
TGWV_CKPT="../models/t=tgwv_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls_cls_m=Hazqeel.electra-small-finetuned-sst2-rotten_tomatoes-distilled/checkpoints/last.ckpt"
# cls_model_name_or_path Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled
python main.py \
 --model_name_or_path facebook/bart-large \
 --num_labels 1 \
 --eval_beams 6 \
 --max_source_length 128 \
 --data_dir ../data/paws \
 --max_target_length 100 \
 --val_max_target_length 100 \
 --test_max_target_length 100 \
 --eval_min_gen_length 80 \
 --eval_max_gen_length 100 \
 --fp16 \
 --do_train \
 --do_eval \
 --num_train_epochs 3 \
 --gradient_accumulation_steps 1 \
 --gpus 1 \
 --eval_batch_size 80 \
 --output_dir ../ckpt/test
# --tgwv_local_ckpt $TGWV_CKPT \

# seq: facebook/bart-large
# cls: Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled