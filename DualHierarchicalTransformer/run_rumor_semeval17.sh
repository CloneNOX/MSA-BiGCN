#!/usr/bin/env bash
for i in 'semeval17'
do
    echo ${i}
    for k in '0'
    do
        echo ${k}
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python run_rumor.py \
        --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 4 --task_name ${i} \
        --output_dir ./output_release/${i}_rumor_output_${k}/ --bert_model bert-base-uncased --do_train --do_eval \
        --max_tweet_num 17 --max_tweet_length 30 --seed 32 --no_cuda
    done
done
