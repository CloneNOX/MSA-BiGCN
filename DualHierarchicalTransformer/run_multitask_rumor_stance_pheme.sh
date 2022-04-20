#!/usr/bin/env bash
for i in 'pheme' # 'twitter15' 'twitter16'
do
    echo ${i}
    for k in '1' '2' '3' '4' '5' '6' '7' '8' # '0' '1' '2' '3' '4' '5' '6' '7' '8' -- '8' '6' '2' '3'  -- '0' '1' '4' '5' '7'
    do
        echo ${k}
        for j in 'DB' # 'FS' 'DB'
        do
            echo ${j}
            PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python run_multitask_rumor_stance.py --data_dir \
            ./rumor_data/${i}/split_${k}/ --data_dir2 ./rumor_data/semeval17_stance/split_0/ --train_batch_size 2 \
            --task_name ${i} --task_name2 semeval17_stance --output_dir ./output_v7/${i}_multitask_output_${j}_${k}/ \
            --bert_model bert-base-uncased --do_eval --mt_model ${j} \
            --max_tweet_num 17 --max_tweet_length 30 --convert_size 100
        done
    done
done
