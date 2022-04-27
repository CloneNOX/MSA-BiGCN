#!/usr/bin/env bash
for i in 'semeval17'
do
    echo ${i}
    for k in '0'
    do
        echo ${k}
        for j in 'DB' # 'FS' 'DB'
        do
            echo ${j}
            PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python run_multitask_rumor_stance.py --data_dir \
            ./rumor_data/${i}/split_${k}/ --data_dir2 ./rumor_data/semeval17_stance/split_0/ --train_batch_size 2 \
            --task_name ${i} --task_name2 semeval17_stance --output_dir ./output_release/${i}_multitask_output_${j}_${k}/ \
            --bert_model bert-base-uncased --do_train --mt_model ${j}\
            --max_tweet_num 17 --max_tweet_length 30 --convert_size 20 --num_train_epochs 2 --no_cuda
            # --do_eval
        done
    done
done

# --do_train --eval_batch_size 4
