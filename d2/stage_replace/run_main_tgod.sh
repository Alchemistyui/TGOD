# ------------------------------------------------------------------------
# training bash to get TGOD format input
# ------------------------------------------------------------------------
# Modified from STAGE (https://github.com/jayleicn/TVQAplus), run_main.sh
# Copyright (c) 2019 Jie Lei. All Rights Reserved
# ------------------------------------------------------------------------
release_path=/DATA_DIR/


debug_vcpt_path=${release_path}/bottom_up_visual_sen_hq_bbt_100_debug.pickle
vfeat_path=${release_path}/tvqa_bbt_bottom_up_pool5_hq_20_100_pca.h5

train_path=${release_path}/tvqa_plus_train_preprocessed.json
valid_path=${release_path}/tvqa_plus_valid_preprocessed.json
test_path=${release_path}/tvqa_plus_test_preprocessed_no_anno.json
qa_bert_path=${release_path}/bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid.h5
sub_bert_path=${release_path}/bbt_sub_s_tokenized_bert_sub_qa_tuned.h5
sub_path=${release_path}/tvqa_plus_subtitles.json

word2idx_path=${release_path}/word2idx.json
eval_object_vocab_path=${release_path}/eval_object_vocab.json
frm_cnt_path=${release_path}/frm_cnt_cache.json

feature_dir=/USR_DIR/datasets/tvqa+/extracted_feature/tgod_pos_tag
train_feature_path=${feature_dir}/tvqa_tensor_feature_train.pt
train_vcpt_path=${feature_dir}/tvqa_vcpt_train.pt
val_feature_path=${feature_dir}/tvqa_tensor_feature_val.pt
val_vcpt_path=${feature_dir}/tvqa_vcpt_val.pt
test_feature_path=${feature_dir}/tvqa_tensor_feature_test.pt
test_vcpt_path=${feature_dir}/tvqa_vcpt_test.pt

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 main_tgod_dist.py \
--train_path ${train_path} \
--valid_path ${valid_path} \
--sub_path ${sub_path} \
--qa_bert_path ${qa_bert_path} \
--sub_bert_path ${sub_bert_path} \
--word2idx_path ${word2idx_path} \
--eval_object_vocab_path ${eval_object_vocab_path} \
--frm_cnt_path ${frm_cnt_path} \
--train_feature_path ${train_feature_path} \
--val_feature_path ${val_feature_path} \
--test_feature_path ${test_feature_path} \
--results_dir_base 'results/tgod' \
--train_vcpt_path ${train_vcpt_path} \
--val_vcpt_path ${val_vcpt_path} \
--test_vcpt_path ${test_vcpt_path} \
--add_local \
--device_ids 0 1 \
--bsz 8 \
--test_bsz 8 \
--use_sup_att \
--log_freq 500 \

