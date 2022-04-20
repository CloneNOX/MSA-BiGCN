#!/usr/bin/env bash
for i in 'semeval17_stance'
do
    echo ${i}
    for k in '0'
    do
        echo ${k}
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python run_multi_stance_semeval17_10BERT.py \
        --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 2 --task_name ${i} \
        --output_dir ./output_release/stance_only_${i}_output10BERT_${k}/ --bert_model bert-base-uncased --do_train --do_eval \
        --max_tweet_num 17 --max_tweet_length 30 --num_train_epochs 27
    done
done
