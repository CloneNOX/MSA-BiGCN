#!/usr/bin/env bash
for i in 'pheme' #'branch'
do
    echo ${i}
    for k in '6' '7' '8' '2' '0' '1' '3' '4' '5' # '0' '1' '2' '3' '4' '5' '6' '7' '8'
    do
        echo ${k}
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=2 python run_rumor.py \
        --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 2 --task_name ${i} \
        --output_dir ./output_v7/${i}_rumor_output_${k}/ --bert_model bert-base-uncased --do_train --do_eval \
        --max_tweet_num 17 --max_tweet_length 30 --no_cuda
    done
done
