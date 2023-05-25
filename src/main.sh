export WANDB_PROJECT="latent-adv"
export CUDA_VISIBLE_DEVICES="4"
export TOKENIZERS_PARALLELISM=false
export USE_TGWV=false
TGWV_CKPT="../models/t=tgwv_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls_cls_m=Hazqeel.electra-small-finetuned-sst2-rotten_tomatoes-distilled/checkpoints/last.ckpt"
# cls_model_name_or_path Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled
python new_main.py \
 --seq2seq_model_name_or_path facebook/bart-large \
 --seq2seq_num_labels 1 \
 --eval_beams 6 \
 --cls_model_name_or_path facebook/bart-large \
 --cls_num_labels 2 \
 --seq2seq_max_source_length 25 \
 --seq2seq_data_dir ../data/yelp/seq2seq \
 --max_target_length 100 \
 --val_max_target_length 100 \
 --test_max_target_length 100 \
 --cls_max_source_length 100 \
 --eval_min_gen_length 80 \
 --eval_max_gen_length 100 \
 --cls_data_dir ../data/rotten_tomatoes/cls \
 --fp16 \
 --do_train_seq2seq \
 --do_eval_seq2seq \
 --save_top_k 20 \
 --num_train_epochs 100 \
 --gradient_accumulation_steps 1 \
 --logger_name wandb \
 --val_metric val_avg_loss \
 --tgwv_v_mode drop \
 --loss_ratio 1.0 \
 --ptb_param embed \
 --attr_dim 0 \
 --seq2seq_model_mode ptb \
 --gpus 1 \
 --vae_loss_ratio 0.3 \
 --limit_val_batches 0.1 \
 --eval_batch_size 80
# --tgwv_local_ckpt $TGWV_CKPT \

# seq: facebook/bart-large
# cls: Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled