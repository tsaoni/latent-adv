export WANDB_PROJECT="latent-adv"
export CUDA_VISIBLE_DEVICES="4"
export TOKENIZERS_PARALLELISM=false
MODEL_DIR="../models"
# TGWV_CKPT="../models/t=tgwv_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls_cls_m=Hazqeel.electra-small-finetuned-sst2-rotten_tomatoes-distilled/checkpoints/last.ckpt"
TGWV_CKPT="../models/softmax_mode/checkpoints/model-epoch=14-val_avg_loss=0.01.ckpt"
SEQ_DIR="t=seq2seq_m=facebook.bart-large_d=..#data#rotten_tomatoes#seq2seq"
SEQ_CKPT="checkpoints/model-epoch=24-val_avg_loss=0.61.ckpt"
LATENT_CKPT="../models/t=cls_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls/latent/85300.pt"

#TGWV_CKPT="../models/drop_mode/checkpoints/model-epoch=06-val_avg_loss=0.11.ckpt"
#TGWV_CKPT="../models/raw_mode/checkpoints/model-epoch=28-val_avg_loss=0.00.ckpt"
# cls_model_name_or_path Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled
python new_main.py \
 --seq2seq_model_name_or_path facebook/bart-large \
 --seq2seq_num_labels 1 \
 --eval_beams 6 \
 --cls_model_name_or_path Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled \
 --cls_num_labels 2 \
 --seq2seq_max_source_length 25 \
 --seq2seq_data_dir ../data/rotten_tomatoes/seq2seq \
 --max_target_length 30 \
 --val_max_target_length 30 \
 --test_max_target_length 30 \
 --cls_max_source_length 25 \
 --eval_min_gen_length 10 \
 --eval_max_gen_length 30 \
 --cls_data_dir ../data/rotten_tomatoes/cls \
 --fp16 \
 --save_top_k 1 \
 --num_train_epochs 100 \
 --gradient_accumulation_steps 1 \
 --logger_name wandb \
 --val_metric val_avg_loss \
 --tgwv_v_mode drop \
 --loss_ratio 1.0 \
 --ptb_param enc_out \
 --attr_dim 0 \
 --seq2seq_model_mode ptb \
 --gpus 1 \
 --vae_loss_ratio 0.3 \
 --do_eval_seq2seq \
 --seq_local_ckpt $MODEL_DIR/$SEQ_DIR/$SEQ_CKPT 
# --latent_ckpt $LATENT_CKPT
 # --seq_local_ckpt $MODEL_DIR/$SEQ_DIR/$SEQ_CKPT 
# --tgwv_local_ckpt $TGWV_CKPT \
# --output_dir $SEQ_DIR \
# seq: facebook/bart-large
# cls: Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled