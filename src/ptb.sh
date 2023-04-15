export WANDB_PROJECT="latent-adv"
export TOKENIZERS_PARALLELISM=false
# TGWV_CKPT="../models/t=tgwv_m=facebook.bart-large_d=..#data#rotten_tomatoes#cls_cls_m=Hazqeel.electra-small-finetuned-sst2-rotten_tomatoes-distilled/checkpoints/last.ckpt"
#TGWV_CKPT="../models/softmax_mode/checkpoints/model-epoch=14-val_avg_loss=0.01.ckpt"
#TGWV_CKPT="../models/drop_mode/checkpoints/model-epoch=06-val_avg_loss=0.11.ckpt"
TGWV_CKPT="../models/raw_mode/checkpoints/model-epoch=28-val_avg_loss=0.00.ckpt"
python main.py \
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
 --output_dir raw_mode \
 --tgwv_v_mode raw \
 --tgwv_local_ckpt $TGWV_CKPT \

# seq: facebook/bart-large
# cls: Hazqeel/electra-small-finetuned-sst2-rotten_tomatoes-distilled