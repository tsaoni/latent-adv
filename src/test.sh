

python model.py \
 --seq2seq_model_name_or_path facebook/bart-large \
 --seq2seq_num_labels 1 \
 --eval_beams 6 \
 --cls_model_name_or_path bert-base-cased \
 --cls_num_labels 2 \
 --seq2seq_max_source_length 10 \
 --seq2seq_data_dir ../data/rotten_tomatoes/seq2seq \
 --max_target_length 10 \
 --val_max_target_length 10 \
 --test_max_target_length 10 \
 --cls_max_source_length 10 \
 --cls_data_dir ../data/rotten_tomatoes/cls 

python dataset.py \
 --seq2seq_max_source_length 10 \
 --seq2seq_data_dir ../data/rotten_tomatoes/seq2seq \
 --max_target_length 10 \
 --val_max_target_length 10 \
 --test_max_target_length 10 \
 --cls_max_source_length 10 \
 --cls_data_dir ../data/rotten_tomatoes/cls 
