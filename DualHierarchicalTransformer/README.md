# Coupled Hierarchical Transformer
Preprocessed SemEval-2017 and PHEME Datasets and Codes for our paper: Coupled Hierarchical Transformer for Stance-Aware Rumor Verification in Social Media Conversations (https://aclanthology.org/2020.emnlp-main.108.pdf).

Author

Jianfei Yu

jfyu@njust.edu.cn

July 15, 2020

## Data (All the datasets are under the "rumor_data" folder)
- Stance Classification (SemEval-2017): the "semeval17_stance" folder.
- Rumor Verification: SemEval-2017 (the "semeval17" folder) & PHEME (the "pheme" folder)


## Requirement

* PyTorch 1.0.0
* Python 3.7

## Code Usage

### Preprocessing
- Step 1: Process the stance classification dataset (SemEval-2017)

```sh
python process_stance_semeval17.py
```

- Step 2: Process the rumor verification dataset (SemEval-2017 & PHEME)

```sh
python process_rumor_semeval17.py
```

```sh
python process_rumor_pheme.py
```

### Model Details
- Hierarchical Transformer for Stance Classification: each conversation thread is split into 10 subthreads, and each subthread is set to have 17 tweets, and each tweet has 30 tokens.
- Hierarchical Transformer and Coupled Hierarchical Transformer for Rumor Verification: Due to memory limitation, each conversation thread is split into 4 subthreads, and each subthread is set to have 17 tweets, and each tweet has 30 tokens.
- Note that for each subthread, you can change the number of tweets and the number of words in each tweet by tuning the two parameters: --max_tweet_num 17 --max_tweet_length 30. Also, it is required that max_tweet_num * max_tweet_length <= 512.


### Training for Stance Classification with Hierarchical Transformer (Single-task Learning)
- SemEval-2017: This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES" based on your available GPUs.

```sh
sh run_multi_stance_semeval17_10BERT.sh
```

### Training for Rumor Verification with Hierarchical Transformer (Single-task Learning)
- SemEval-2017: This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES" based on your available GPUs.

```sh
sh run_rumor_semeval17.sh
```

- PHEME: This is the training code of performing 9 fold cross validation. Note that you can change "CUDA_VISIBLE_DEVICES" based on your available GPUs.

```sh
sh run_rumor_pheme.sh
```

### Training for Rumor Verification with Coupled Hierarchical Transformer (Multi-task Learning)
- SemEval-2017: This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES" based on your available GPUs.

```sh
sh run_multitask_rumor_stance.sh
```

- PHEME: This is the training code of performing 9 fold cross validation. Note that you can change "CUDA_VISIBLE_DEVICES" based on your available GPUs.

```sh
sh run_multitask_rumor_stance_pheme.sh
```

### We show our running logs on the SemEval-2017 dataset in the folder "running_log_files", and you can find similar results as reported in our EMNLP submission.


## Acknowledgements
- Using these two datasets means you have read and accepted the copyrights set by Twitter and dataset providers.
