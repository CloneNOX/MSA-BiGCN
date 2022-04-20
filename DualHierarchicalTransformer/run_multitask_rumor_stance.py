# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from sequence_labeling import classification_report

from my_bert.tokenization import BertTokenizer
from my_bert.modeling import FullySharedBert, DualBert
from my_bert.optimization import BertAdam
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from sklearn.metrics import precision_recall_fscore_support


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2,
                 input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, input_mask,
                 label_id, stance_position, label_mask):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.input_ids3 = input_ids3
        self.input_mask3 = input_mask3
        self.segment_ids3 = segment_ids3
        self.input_ids4 = input_ids4
        self.input_mask4 = input_mask4
        self.segment_ids4 = segment_ids4
        self.input_mask = input_mask
        self.label_id = label_id
        self.stance_position = stance_position
        self.label_mask = label_mask

class InputStanceFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2,
                 input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, input_mask,
                 label_id, stance_position, label_mask):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.input_ids3 = input_ids3
        self.input_mask3 = input_mask3
        self.segment_ids3 = segment_ids3
        self.input_ids4 = input_ids4
        self.input_mask4 = input_mask4
        self.segment_ids4 = segment_ids4
        self.input_mask = input_mask
        self.label_id = label_id
        self.stance_position = stance_position
        self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class RumorProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2].lower().split('|||||')
            text_b = None
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StanceProcessor(DataProcessor):
    """Processor for the Stance Prediction data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "stance_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["B-DENY", "B-SUPPORT", "B-QUERY", "B-COMMENT"]
        # should be ["B-DENY", "B-SUPPORT", "B-QUERY", "B-COMMENT"], but it does not matter, and we can solve this
        # at the evaluation stage

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2].lower().split('|||||')
            text_b = None
            label = line[1].split(',')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_tweet_num, max_tweet_len):
    """Loads a data file into a list of `InputBatch`s."""

    #max_tweet_len = 20  # the number of words in each tweet (42,12) for rumor verification
    #max_tweet_num = 25  # the number of tweets in each bucket
    max_bucket_num = 4  # the
    # number of buckets in each thread

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tweetlist = example.text_a
        #tweetlist = tweetlist[:max_tweet_num]
        #labellist = example.label[:max_tweet_num]
        label = example.label

        tweets_tokens = []
        for i, cur_tweet in enumerate(tweetlist):
            tweet = tweetlist[i]
            if tweet == '':
                break
            tweet_token = tokenizer.tokenize(tweet)
            if len(tweet_token) >= max_tweet_len - 1:
                tweet_token = tweet_token[:(max_tweet_len - 2)]
            tweets_tokens.append(tweet_token)

        if len(tweets_tokens) <= max_tweet_num:
            tweets_tokens1 = tweets_tokens
            tweets_tokens2, tweets_tokens3, tweets_tokens4 = [], [], []
        elif len(tweets_tokens) > max_tweet_num and len(tweets_tokens) <= max_tweet_num*2:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:]
            tweets_tokens3, tweets_tokens4 = [], []
        elif len(tweets_tokens) > max_tweet_num*2 and len(tweets_tokens) <= max_tweet_num*3:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:]
            tweets_tokens4 = []
        elif len(tweets_tokens) > max_tweet_num*3 and len(tweets_tokens) <= max_tweet_num*4:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:]
        else:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]

        input_tokens1, input_ids1, input_mask1, segment_ids1, stance_position1, label_mask1 = \
            bucket_rumor_conversion(tweets_tokens1, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens2, input_ids2, input_mask2, segment_ids2, stance_position2, label_mask2 = \
            bucket_rumor_conversion(tweets_tokens2, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens3, input_ids3, input_mask3, segment_ids3, stance_position3, label_mask3 = \
            bucket_rumor_conversion(tweets_tokens3, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens4, input_ids4, input_mask4, segment_ids4, stance_position4, label_mask4 = \
            bucket_rumor_conversion(tweets_tokens4, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)

        stance_position =[]
        stance_position.extend(stance_position1)
        stance_position.extend(stance_position2)
        stance_position.extend(stance_position3)
        stance_position.extend(stance_position4)
        input_mask = []
        input_mask.extend(input_mask1)
        input_mask.extend(input_mask2)
        input_mask.extend(input_mask3)
        input_mask.extend(input_mask4)
        label_mask = []
        label_mask.extend(label_mask1)
        label_mask.extend(label_mask2)
        label_mask.extend(label_mask3)
        label_mask.extend(label_mask4)

        label_id = label_map[example.label]

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in input_tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids1]))
            logger.info("label: %s" % (label))

        features.append(
            InputFeatures(input_ids1=input_ids1, input_mask1=input_mask1, segment_ids1=segment_ids1,
                          input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2,
                          input_ids3=input_ids3, input_mask3=input_mask3, segment_ids3=segment_ids3,
                          input_ids4=input_ids4, input_mask4=input_mask4, segment_ids4=segment_ids4,
                          input_mask=input_mask, label_id=label_id, stance_position=stance_position, label_mask=label_mask))
    return features


def bucket_rumor_conversion(tweets_tokens, tokenizer, max_tweet_num, max_tweet_len, max_seq_length):
    ntokens = []
    input_tokens = []
    input_ids = []
    input_mask = []
    segment_ids = []
    stance_position = []
    label_mask = []
    if tweets_tokens != []:
        ntokens.append("[CLS]")
        # input_tokens.extend(ntokens) # avoid having two [CLS] at the begining
        # segment_ids.append(0) #########no need to add this line
        stance_position.append(0)
        label_mask.append(1)
    for i, tweet_token in enumerate(tweets_tokens):
        if i != 0:
            ntokens = []
            ntokens.append("[CLS]")
            stance_position.append(len(input_ids))
            label_mask.append(1)
        ntokens.extend(tweet_token)
        ntokens.append("[SEP]")
        input_tokens.extend(ntokens)  # just for printing out
        input_tokens.extend("[padpadpad]")  # just for printing out
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        while len(tweet_input_ids) < max_tweet_len:
            tweet_input_ids.append(0)
            tweet_input_mask.append(0)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [i % 2] * len(tweet_input_ids)

    cur_tweet_num = len(tweets_tokens)
    pad_tweet_length = max_tweet_num - cur_tweet_num
    for j in range(pad_tweet_length):
        ntokens = []
        ntokens.append("[CLS]")
        ntokens.append("[SEP]")
        stance_position.append(len(input_ids))
        label_mask.append(0)
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        tweet_input_ids = tweet_input_ids + [0] * (max_tweet_len - 2)
        tweet_input_mask = tweet_input_mask + [0] * (max_tweet_len - 2)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [(cur_tweet_num + j) % 2] * max_tweet_len

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_tokens, input_ids, input_mask, segment_ids, stance_position, label_mask


def bucket_conversion(tweets_tokens, labels, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length):
    ntokens = []
    input_tokens = []
    input_ids = []
    input_mask = []
    segment_ids = []
    label_ids = []
    label_mask = []
    stance_position = []
    if labels != []:
        ntokens.append("[CLS]")
        # input_tokens.extend(ntokens) # avoid having two [CLS] at the begining
        # segment_ids.append(0) #########no need to add this line
        label_ids.append(label_map[labels[0]])
        stance_position.append(0)
        label_mask.append(1)
    for i, tweet_token in enumerate(tweets_tokens):
        if i != 0:
            ntokens = []
            ntokens.append("[CLS]")
            label_ids.append(label_map[labels[i]])
            stance_position.append(len(input_ids))
            label_mask.append(1)
        ntokens.extend(tweet_token)
        ntokens.append("[SEP]")
        input_tokens.extend(ntokens)  # just for printing out
        input_tokens.extend("[padpadpad]")  # just for printing out
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        while len(tweet_input_ids) < max_tweet_len:
            tweet_input_ids.append(0)
            tweet_input_mask.append(0)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [i % 2] * len(tweet_input_ids)

    cur_tweet_num = len(tweets_tokens)
    pad_tweet_length = max_tweet_num - cur_tweet_num
    for j in range(pad_tweet_length):
        ntokens = []
        ntokens.append("[CLS]")
        ntokens.append("[SEP]")
        label_ids.append(0)
        stance_position.append(len(input_ids))
        label_mask.append(0)
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        tweet_input_ids = tweet_input_ids + [0] * (max_tweet_len - 2)
        tweet_input_mask = tweet_input_mask + [0] * (max_tweet_len - 2)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        segment_ids = segment_ids + [(cur_tweet_num + j) % 2] * max_tweet_len

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_tokens, input_ids, input_mask, segment_ids, label_ids, stance_position, label_mask


def convert_stance_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_tweet_num, max_tweet_len):
    """Loads a data file into a list of `InputBatch`s."""

    #max_tweet_len = 20  # the number of words in each tweet (42,12) performs best
    #max_tweet_num = 25  # the number of tweets in each bucket
    max_bucket_num = 4  # the number of buckets in each thread
    label_map_dict = {'0':'B-DENY', '1':'B-SUPPORT', '2':'B-QUERY', '3':'B-COMMENT'}

    label_map = {label : i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tweetlist = example.text_a
        #tweetlist = tweetlist[:max_tweet_num]
        #labellist = example.label[:max_tweet_num]
        labellist = example.label[:max_tweet_num*max_bucket_num]

        tweets_tokens = []
        labels = []
        for i, label in enumerate(labellist):
            tweet = tweetlist[i]
            if tweet == '':
                break
            tweet_token = tokenizer.tokenize(tweet)
            if len(tweet_token) >= max_tweet_len - 1:
                tweet_token = tweet_token[:(max_tweet_len - 2)]
            tweets_tokens.append(tweet_token)
            label_1 = label
            labels.append(label_map_dict[label_1])

        if len(labels) <= max_tweet_num:
            tweets_tokens1 = tweets_tokens
            labels1 = labels
            tweets_tokens2, labels2, tweets_tokens3, labels3, tweets_tokens4, labels4 = [], [], [], [], [], []
        elif len(labels) > max_tweet_num and len(labels) <= max_tweet_num*2:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:]
            labels2 = labels[max_tweet_num:]
            tweets_tokens3, labels3, tweets_tokens4, labels4 = [], [], [], []
        elif len(labels) > max_tweet_num*2 and len(labels) <= max_tweet_num*3:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:]
            labels3 = labels[max_tweet_num*2:]
            tweets_tokens4, labels4 = [], []
        elif len(labels) > max_tweet_num*3 and len(labels) <= max_tweet_num*4:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:]
            labels4 = labels[max_tweet_num*3:]
        else:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]

        input_tokens1, input_ids1, input_mask1, segment_ids1, label_ids1, stance_position1, label_mask1 = \
            bucket_conversion(tweets_tokens1, labels1, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens2, input_ids2, input_mask2, segment_ids2, label_ids2, stance_position2, label_mask2 = \
            bucket_conversion(tweets_tokens2, labels2, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens3, input_ids3, input_mask3, segment_ids3, label_ids3, stance_position3, label_mask3 = \
            bucket_conversion(tweets_tokens3, labels3, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens4, input_ids4, input_mask4, segment_ids4, label_ids4, stance_position4, label_mask4 = \
            bucket_conversion(tweets_tokens4, labels4, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)

        label_ids = []
        label_ids.extend(label_ids1)
        label_ids.extend(label_ids2)
        label_ids.extend(label_ids3)
        label_ids.extend(label_ids4)
        stance_position =[]
        stance_position.extend(stance_position1)
        stance_position.extend(stance_position2)
        stance_position.extend(stance_position3)
        stance_position.extend(stance_position4)
        label_mask = []
        label_mask.extend(label_mask1)
        label_mask.extend(label_mask2)
        label_mask.extend(label_mask3)
        label_mask.extend(label_mask4)
        input_mask = []
        input_mask.extend(input_mask1)
        input_mask.extend(input_mask2)
        input_mask.extend(input_mask3)
        input_mask.extend(input_mask4)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in input_tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids1]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids1]))

        features.append(
            InputStanceFeatures(input_ids1=input_ids1, input_mask1=input_mask1, segment_ids1=segment_ids1,
                                input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2,
                                input_ids3=input_ids3, input_mask3=input_mask3, segment_ids3=segment_ids3,
                                input_ids4=input_ids4, input_mask4=input_mask4, segment_ids4=segment_ids4,
                          input_mask=input_mask, label_id=label_ids, stance_position=stance_position, label_mask=label_mask))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def rumor_macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='../absa_data/twitter',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir2",
                        default='../absa_data/twitter',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='twitter',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--task_name2",
                        default='pheme',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--bertlayer', action='store_true', help='whether to add another bert layer')
    parser.add_argument('--mt_model', default='FS', help='model name')  # 'FS', 'SS', 'DB'
    parser.add_argument('--max_tweet_num', type=int, default=30, help="the maximum number of tweets")
    parser.add_argument('--max_tweet_length', type=int, default=17, help="the maximum length of each tweet")
    parser.add_argument('--convert_size', type=int, default=20, help="conversion size")

    args = parser.parse_args()

    if args.bertlayer:
        print("add another bert layer")
    else:
        print("pre-trained bert without additional bert layer")

    if args.task_name.lower() == "semeval17":
        if args.mt_model == 'FS':
            args.num_train_epochs = 26
        else:
            args.num_train_epochs = 29 # 25,30

    if args.task_name.lower() == "pheme":
        if args.mt_model == 'FS':
            args.seed = 16 # other folds: 5, fold 3 & 8: 16
            args.num_train_epochs = 7
        else:
            args.seed = 16  # 42: very good, 2: good, 64, relatively good
            args.num_train_epochs = 7

    if args.task_name.lower() == "pheme5":
        args.num_train_epochs = 7

    processors = {
        "semeval17": RumorProcessor,
        "pheme": RumorProcessor,
        "pheme5": RumorProcessor,
        "semeval17_stance": StanceProcessor
    }

    num_labels_task = {
        "semeval17": 3,
        "pheme": 3,
        "pheme5": 3
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    task_name2 = args.task_name2.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    if task_name2 not in processors:
        raise ValueError("Task 2 not found: %s" % (task_name2))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = num_labels_task[task_name] # label 0 corresponds to padding, label in label_list starts from 1

    stance_processor = processors[task_name2]()
    stance_label_list = stance_processor.get_labels()
    stance_num_labels = len(stance_label_list) + 1 # label 0 corresponds to padding, label in label_list starts from 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    extended_stance_train_examples = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        stance_train_examples = stance_processor.get_train_examples(args.data_dir2)
        # enrich stance_train_examples to be the same as train examples
        src_tgt_ratio = int(len(train_examples) / len(stance_train_examples))
        remaining_samp_len = len(train_examples)-src_tgt_ratio*len(stance_train_examples)
        extended_stance_train_examples = stance_train_examples*src_tgt_ratio
        extended_stance_train_examples.extend(stance_train_examples[:remaining_samp_len])

        assert len(train_examples) == len(extended_stance_train_examples)

    # Prepare model
    if args.mt_model == 'FS':
        print('The current multi-task learning model is FS ...')
        model = FullySharedBert.from_pretrained(args.bert_model,
                                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                    args.local_rank),
                                                rumor_num_labels=num_labels, stance_num_labels=stance_num_labels,
                                                max_tweet_num=args.max_tweet_num, max_tweet_length=args.max_tweet_length,
                                                convert_size=args.convert_size)
    elif args.mt_model == 'DB':
        print('The current multi-task learning model is our Dual Bert model...')
        model = DualBert.from_pretrained(args.bert_model,
                                         cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                    args.local_rank),
                                         rumor_num_labels=num_labels, stance_num_labels=stance_num_labels,
                                         max_tweet_num=args.max_tweet_num, max_tweet_length=args.max_tweet_length,
                                         convert_size=args.convert_size)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    stance_optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        print('training data')
        # rumor detection task
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.max_tweet_num, args.max_tweet_length)

        all_input_ids1 = torch.tensor([f.input_ids1 for f in train_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask1 for f in train_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids1 for f in train_features], dtype=torch.long)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype=torch.long)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype=torch.long)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype=torch.long)
        all_input_ids3 = torch.tensor([f.input_ids3 for f in train_features], dtype=torch.long)
        all_input_mask3 = torch.tensor([f.input_mask3 for f in train_features], dtype=torch.long)
        all_segment_ids3 = torch.tensor([f.segment_ids3 for f in train_features], dtype=torch.long)
        all_input_ids4 = torch.tensor([f.input_ids4 for f in train_features], dtype=torch.long)
        all_input_mask4 = torch.tensor([f.input_mask4 for f in train_features], dtype=torch.long)
        all_segment_ids4 = torch.tensor([f.segment_ids4 for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)

        # stance classification task
        stance_train_features = convert_stance_examples_to_features(extended_stance_train_examples,
            stance_label_list, args.max_seq_length, tokenizer, args.max_tweet_num, args.max_tweet_length)

        stance_all_input_ids1 = torch.tensor([f.input_ids1 for f in stance_train_features], dtype=torch.long)
        stance_all_input_mask1 = torch.tensor([f.input_mask1 for f in stance_train_features], dtype=torch.long)
        stance_all_segment_ids1 = torch.tensor([f.segment_ids1 for f in stance_train_features], dtype=torch.long)
        stance_all_input_ids2 = torch.tensor([f.input_ids2 for f in stance_train_features], dtype=torch.long)
        stance_all_input_mask2 = torch.tensor([f.input_mask2 for f in stance_train_features], dtype=torch.long)
        stance_all_segment_ids2 = torch.tensor([f.segment_ids2 for f in stance_train_features], dtype=torch.long)
        stance_all_input_ids3 = torch.tensor([f.input_ids3 for f in stance_train_features], dtype=torch.long)
        stance_all_input_mask3 = torch.tensor([f.input_mask3 for f in stance_train_features], dtype=torch.long)
        stance_all_segment_ids3 = torch.tensor([f.segment_ids3 for f in stance_train_features], dtype=torch.long)
        stance_all_input_ids4 = torch.tensor([f.input_ids4 for f in stance_train_features], dtype=torch.long)
        stance_all_input_mask4 = torch.tensor([f.input_mask4 for f in stance_train_features], dtype=torch.long)
        stance_all_segment_ids4 = torch.tensor([f.segment_ids4 for f in stance_train_features], dtype=torch.long)
        stance_all_input_mask = torch.tensor([f.input_mask for f in stance_train_features], dtype=torch.long)
        stance_all_label_ids = torch.tensor([f.label_id for f in stance_train_features], dtype=torch.long)
        #stance_all_stance_position = torch.tensor([f.stance_position for f in stance_train_features], dtype=torch.long)
        stance_all_label_mask = torch.tensor([f.label_mask for f in stance_train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                   all_input_ids2, all_input_mask2, all_segment_ids2,
                                   all_input_ids3, all_input_mask3, all_segment_ids3,
                                   all_input_ids4, all_input_mask4, all_segment_ids4,
                                   all_input_mask, all_label_ids, all_label_mask,
                                   stance_all_input_ids1, stance_all_input_mask1, stance_all_segment_ids1,
                                   stance_all_input_ids2, stance_all_input_mask2, stance_all_segment_ids2,
                                   stance_all_input_ids3, stance_all_input_mask3, stance_all_segment_ids3,
                                   stance_all_input_ids4, stance_all_input_mask4, stance_all_segment_ids4,
                                   stance_all_input_mask, stance_all_label_ids, stance_all_label_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        #''' for dev evaluation
        # rumor detection task
        eval_examples = processor.get_dev_examples(args.data_dir)
        print('dev data')
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_tweet_num, args.max_tweet_length)

        all_input_ids1 = torch.tensor([f.input_ids1 for f in eval_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask1 for f in eval_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids1 for f in eval_features], dtype=torch.long)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in eval_features], dtype=torch.long)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in eval_features], dtype=torch.long)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in eval_features], dtype=torch.long)
        all_input_ids3 = torch.tensor([f.input_ids3 for f in eval_features], dtype=torch.long)
        all_input_mask3 = torch.tensor([f.input_mask3 for f in eval_features], dtype=torch.long)
        all_segment_ids3 = torch.tensor([f.segment_ids3 for f in eval_features], dtype=torch.long)
        all_input_ids4 = torch.tensor([f.input_ids4 for f in eval_features], dtype=torch.long)
        all_input_mask4 = torch.tensor([f.input_mask4 for f in eval_features], dtype=torch.long)
        all_segment_ids4 = torch.tensor([f.segment_ids4 for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)

        # stance classification task
        stance_eval_examples = stance_processor.get_test_examples(args.data_dir2)
        # enrich stance_train_examples to be the same as train examples
        eval_src_tgt_ratio = int(len(eval_examples) / len(stance_eval_examples))
        eval_remaining_samp_len = len(eval_examples)-eval_src_tgt_ratio*len(stance_eval_examples)
        extended_stance_eval_examples = stance_eval_examples*eval_src_tgt_ratio
        extended_stance_eval_examples.extend(stance_eval_examples[:eval_remaining_samp_len])
        stance_eval_features = convert_stance_examples_to_features(extended_stance_eval_examples,
                                                                   stance_label_list, args.max_seq_length, tokenizer,
                                                                   args.max_tweet_num, args.max_tweet_length)

        stance_all_input_ids1 = torch.tensor([f.input_ids1 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask1 = torch.tensor([f.input_mask1 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids1 = torch.tensor([f.segment_ids1 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_ids2 = torch.tensor([f.input_ids2 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask2 = torch.tensor([f.input_mask2 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids2 = torch.tensor([f.segment_ids2 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_ids3 = torch.tensor([f.input_ids3 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask3 = torch.tensor([f.input_mask3 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids3 = torch.tensor([f.segment_ids3 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_ids4 = torch.tensor([f.input_ids4 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask4 = torch.tensor([f.input_mask4 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids4 = torch.tensor([f.segment_ids4 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask = torch.tensor([f.input_mask for f in stance_eval_features], dtype=torch.long)
        stance_all_label_ids = torch.tensor([f.label_id for f in stance_eval_features], dtype=torch.long)
        #stance_all_stance_position = torch.tensor([f.stance_position for f in stance_eval_features], dtype=torch.long)
        stance_all_label_mask = torch.tensor([f.label_mask for f in stance_eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                  all_input_ids2, all_input_mask2, all_segment_ids2,
                                  all_input_ids3, all_input_mask3, all_segment_ids3,
                                  all_input_ids4, all_input_mask4, all_segment_ids4,
                                  all_input_mask, all_label_ids, all_label_mask,
                                  stance_all_input_ids1, stance_all_input_mask1, stance_all_segment_ids1,
                                  stance_all_input_ids2, stance_all_input_mask2, stance_all_segment_ids2,
                                  stance_all_input_ids3, stance_all_input_mask3, stance_all_segment_ids3,
                                  stance_all_input_ids4, stance_all_input_mask4, stance_all_segment_ids4,
                                  stance_all_input_mask, stance_all_label_ids, stance_all_label_mask
                                  )
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        #''' for test evaluation
        # rumor detection task
        test_examples = processor.get_test_examples(args.data_dir)
        print('dev data')
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, args.max_tweet_num, args.max_tweet_length)

        all_input_ids1 = torch.tensor([f.input_ids1 for f in test_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask1 for f in test_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids1 for f in test_features], dtype=torch.long)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in test_features], dtype=torch.long)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in test_features], dtype=torch.long)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in test_features], dtype=torch.long)
        all_input_ids3 = torch.tensor([f.input_ids3 for f in test_features], dtype=torch.long)
        all_input_mask3 = torch.tensor([f.input_mask3 for f in test_features], dtype=torch.long)
        all_segment_ids3 = torch.tensor([f.segment_ids3 for f in test_features], dtype=torch.long)
        all_input_ids4 = torch.tensor([f.input_ids4 for f in test_features], dtype=torch.long)
        all_input_mask4 = torch.tensor([f.input_mask4 for f in test_features], dtype=torch.long)
        all_segment_ids4 = torch.tensor([f.segment_ids4 for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)

        # stance classification task
        stance_eval_examples = stance_processor.get_test_examples(args.data_dir2)
        # enrich stance_train_examples to be the same as train examples
        eval_src_tgt_ratio = int(len(test_examples) / len(stance_eval_examples))
        eval_remaining_samp_len = len(test_examples)-eval_src_tgt_ratio*len(stance_eval_examples)
        extended_stance_eval_examples = stance_eval_examples*eval_src_tgt_ratio
        extended_stance_eval_examples.extend(stance_eval_examples[:eval_remaining_samp_len])
        stance_eval_features = convert_stance_examples_to_features(extended_stance_eval_examples,
                                                                   stance_label_list, args.max_seq_length, tokenizer,
                                                                   args.max_tweet_num, args.max_tweet_length)

        stance_all_input_ids1 = torch.tensor([f.input_ids1 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask1 = torch.tensor([f.input_mask1 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids1 = torch.tensor([f.segment_ids1 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_ids2 = torch.tensor([f.input_ids2 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask2 = torch.tensor([f.input_mask2 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids2 = torch.tensor([f.segment_ids2 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_ids3 = torch.tensor([f.input_ids3 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask3 = torch.tensor([f.input_mask3 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids3 = torch.tensor([f.segment_ids3 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_ids4 = torch.tensor([f.input_ids4 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask4 = torch.tensor([f.input_mask4 for f in stance_eval_features], dtype=torch.long)
        stance_all_segment_ids4 = torch.tensor([f.segment_ids4 for f in stance_eval_features], dtype=torch.long)
        stance_all_input_mask = torch.tensor([f.input_mask for f in stance_eval_features], dtype=torch.long)
        stance_all_label_ids = torch.tensor([f.label_id for f in stance_eval_features], dtype=torch.long)
        #stance_all_stance_position = torch.tensor([f.stance_position for f in stance_eval_features], dtype=torch.long)
        stance_all_label_mask = torch.tensor([f.label_mask for f in stance_eval_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                  all_input_ids2, all_input_mask2, all_segment_ids2,
                                  all_input_ids3, all_input_mask3, all_segment_ids3,
                                  all_input_ids4, all_input_mask4, all_segment_ids4,
                                  all_input_mask, all_label_ids, all_label_mask,
                                  stance_all_input_ids1, stance_all_input_mask1, stance_all_segment_ids1,
                                  stance_all_input_ids2, stance_all_input_mask2, stance_all_segment_ids2,
                                  stance_all_input_ids3, stance_all_input_mask3, stance_all_segment_ids3,
                                  stance_all_input_ids4, stance_all_input_mask4, stance_all_segment_ids4,
                                  stance_all_input_mask, stance_all_label_ids, stance_all_label_mask
                                  )
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)


        max_acc_f1 = 0.0
        logger.info("*************** Running training ***************")
        '''
        if args.task_name.lower() == "semeval17":
            logger.info("********** Pre-training Stance **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()
            stance_tr_loss = 0
            stance_nb_tr_examples, stance_nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_mask, label_ids, label_mask, stance_input_ids1, stance_input_mask1, stance_segment_ids1, \
                stance_input_ids2, stance_input_mask2, stance_segment_ids2, stance_input_ids3, stance_input_mask3, \
                stance_segment_ids3, stance_input_ids4, stance_input_mask4, stance_segment_ids4, \
                stance_input_mask, stance_label_ids, stance_label_mask = batch

                # optimize stance classification task
                stance_loss = model(stance_input_ids1, stance_segment_ids1, stance_input_mask1, stance_input_ids2,
                                    stance_segment_ids2, stance_input_mask2, stance_input_ids3, stance_segment_ids3,
                                    stance_input_mask3, stance_input_ids4, stance_segment_ids4, stance_input_mask4,
                                    stance_input_mask,
                                    task='stance', stance_labels=stance_label_ids, stance_label_mask=stance_label_mask)
                if n_gpu > 1:
                    stance_loss = stance_loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    stance_loss = stance_loss / args.gradient_accumulation_steps

                if args.fp16:
                    stance_optimizer.backward(stance_loss)
                else:
                    stance_loss.backward()

                stance_tr_loss += stance_loss.item()
                stance_nb_tr_examples += stance_input_ids1.size(0)
                stance_nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    stance_lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                             args.warmup_proportion)
                    for param_group in stance_optimizer.param_groups:
                        param_group['lr'] = stance_lr_this_step
                    stance_optimizer.step()
                    stance_optimizer.zero_grad()
        '''
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):

            logger.info("********** Epoch: "+ str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()
            tr_loss = 0
            stance_tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            stance_nb_tr_examples, stance_nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_mask, label_ids, label_mask, stance_input_ids1, stance_input_mask1, stance_segment_ids1, \
                stance_input_ids2, stance_input_mask2, stance_segment_ids2, stance_input_ids3, stance_input_mask3, \
                stance_segment_ids3, stance_input_ids4, stance_input_mask4, stance_segment_ids4, \
                stance_input_mask, stance_label_ids, stance_label_mask = batch

                # optimize rumor detection task
                loss = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                             input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                             input_mask, rumor_labels=label_ids, stance_label_mask=label_mask)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids1.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # optimize stance classification task
                stance_loss = model(stance_input_ids1, stance_segment_ids1, stance_input_mask1, stance_input_ids2,
                                    stance_segment_ids2, stance_input_mask2, stance_input_ids3, stance_segment_ids3,
                                    stance_input_mask3, stance_input_ids4, stance_segment_ids4, stance_input_mask4,
                                    stance_input_mask,
                                    task='stance', stance_labels=stance_label_ids,
                                    stance_label_mask=stance_label_mask)
                if n_gpu > 1:
                    stance_loss = stance_loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    stance_loss = stance_loss / args.gradient_accumulation_steps

                if args.fp16:
                    stance_optimizer.backward(stance_loss)
                else:
                    stance_loss.backward()

                stance_tr_loss += stance_loss.item()
                stance_nb_tr_examples += stance_input_ids1.size(0)
                stance_nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    stance_lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                             args.warmup_proportion)
                    for param_group in stance_optimizer.param_groups:
                        param_group['lr'] = stance_lr_this_step
                    stance_optimizer.step()
                    stance_optimizer.zero_grad()

            logger.info("\n************************************************** Running evaluation on Dev Set****************************************")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            stance_y_true = []
            stance_y_pred = []

            stance_label_map = {i: label for i, label in enumerate(stance_label_list, 1)}
            stance_label_map[0] = "PAD"

            for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_mask, label_ids, label_mask, stance_input_ids1, stance_input_mask1, stance_segment_ids1, \
                stance_input_ids2, stance_input_mask2, stance_segment_ids2, stance_input_ids3, stance_input_mask3, \
                stance_segment_ids3, stance_input_ids4, stance_input_mask4, stance_segment_ids4, \
                stance_input_mask, stance_label_ids, stance_label_mask in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids1 = input_ids1.to(device)
                input_mask1 = input_mask1.to(device)
                segment_ids1 = segment_ids1.to(device)
                input_ids2 = input_ids2.to(device)
                input_mask2 = input_mask2.to(device)
                segment_ids2 = segment_ids2.to(device)
                input_ids3 = input_ids3.to(device)
                input_mask3 = input_mask3.to(device)
                segment_ids3 = segment_ids3.to(device)
                input_ids4 = input_ids4.to(device)
                input_mask4 = input_mask4.to(device)
                segment_ids4 = segment_ids4.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)
                label_mask = label_mask.to(device)

                stance_input_ids1 = stance_input_ids1.to(device)
                stance_input_mask1 = stance_input_mask1.to(device)
                stance_segment_ids1 = stance_segment_ids1.to(device)
                stance_input_ids2 = stance_input_ids2.to(device)
                stance_input_mask2 = stance_input_mask2.to(device)
                stance_segment_ids2 = stance_segment_ids2.to(device)
                stance_input_ids3 = stance_input_ids3.to(device)
                stance_input_mask3 = stance_input_mask3.to(device)
                stance_segment_ids3 = stance_segment_ids3.to(device)
                stance_input_ids4 = stance_input_ids4.to(device)
                stance_input_mask4 = stance_input_mask4.to(device)
                stance_segment_ids4 = stance_segment_ids4.to(device)
                stance_input_mask = stance_input_mask.to(device)
                stance_label_ids = stance_label_ids.to(device)
                #stance_stance_position = stance_stance_position.to(device)
                stance_label_mask = stance_label_mask.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_mask, label_ids, stance_label_mask=label_mask)
                    logits, _ = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_mask, stance_label_mask=label_mask)
                    stance_logits = model(stance_input_ids1, stance_segment_ids1, stance_input_mask1, stance_input_ids2,
                                    stance_segment_ids2, stance_input_mask2, stance_input_ids3, stance_segment_ids3,
                                    stance_input_mask3, stance_input_ids4, stance_segment_ids4, stance_input_mask4,
                                    stance_input_mask, task='stance', stance_label_mask=stance_label_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                stance_logits = torch.argmax(F.log_softmax(stance_logits, dim=2), dim=2)
                stance_logits = stance_logits.detach().cpu().numpy()
                stance_label_ids = stance_label_ids.to('cpu').numpy()
                stance_label_mask = stance_label_mask.to('cpu').numpy()
                for i, mask in enumerate(stance_label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            temp_1.append(stance_label_map[stance_label_ids[i][j]])
                            temp_2.append(stance_label_map[stance_logits[i][j]])
                        else:
                            break
                    stance_y_true.append(temp_1)
                    stance_y_pred.append(temp_2)

                nb_eval_examples += input_ids1.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = rumor_macro_f1(true_label, pred_outputs)

            stance_report = classification_report(stance_y_true, stance_y_pred, digits=4)
            logger.info("\n***** Dev Stance Eval results *****")
            logger.info("\n%s", stance_report)
            stance_eval_true_label = np.concatenate(stance_y_true)
            stance_eval_pred_label = np.concatenate(stance_y_pred)
            stance_precision, stance_recall, stance_F_score = macro_f1(stance_eval_true_label, stance_eval_pred_label)
            print("stance_F-score: ", stance_F_score)

            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'f_score': F_score,
                      'global_step': global_step,
                      'loss': loss}

            logger.info("\n***** Dev Rumor Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if eval_accuracy+F_score > max_acc_f1:   # if eval_accuracy+F_score+stance_F_score >= max_acc_f1:
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                if args.do_train:
                    torch.save(model_to_save.state_dict(), output_model_file)
                max_acc_f1 = eval_accuracy+F_score
                #max_acc_f1 = eval_accuracy+F_score+stance_F_score


            logger.info("\n************************************************** Running evaluation on Test Set****************************************")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            stance_y_true = []
            stance_y_pred = []

            stance_label_map = {i: label for i, label in enumerate(stance_label_list, 1)}
            stance_label_map[0] = "PAD"

            for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_mask, label_ids, label_mask, stance_input_ids1, stance_input_mask1, stance_segment_ids1, \
                stance_input_ids2, stance_input_mask2, stance_segment_ids2, stance_input_ids3, stance_input_mask3, \
                stance_segment_ids3, stance_input_ids4, stance_input_mask4, stance_segment_ids4, \
                stance_input_mask, stance_label_ids, stance_label_mask in tqdm(test_dataloader, desc="Evaluating"):
                input_ids1 = input_ids1.to(device)
                input_mask1 = input_mask1.to(device)
                segment_ids1 = segment_ids1.to(device)
                input_ids2 = input_ids2.to(device)
                input_mask2 = input_mask2.to(device)
                segment_ids2 = segment_ids2.to(device)
                input_ids3 = input_ids3.to(device)
                input_mask3 = input_mask3.to(device)
                segment_ids3 = segment_ids3.to(device)
                input_ids4 = input_ids4.to(device)
                input_mask4 = input_mask4.to(device)
                segment_ids4 = segment_ids4.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)
                label_mask = label_mask.to(device)

                stance_input_ids1 = stance_input_ids1.to(device)
                stance_input_mask1 = stance_input_mask1.to(device)
                stance_segment_ids1 = stance_segment_ids1.to(device)
                stance_input_ids2 = stance_input_ids2.to(device)
                stance_input_mask2 = stance_input_mask2.to(device)
                stance_segment_ids2 = stance_segment_ids2.to(device)
                stance_input_ids3 = stance_input_ids3.to(device)
                stance_input_mask3 = stance_input_mask3.to(device)
                stance_segment_ids3 = stance_segment_ids3.to(device)
                stance_input_ids4 = stance_input_ids4.to(device)
                stance_input_mask4 = stance_input_mask4.to(device)
                stance_segment_ids4 = stance_segment_ids4.to(device)
                stance_input_mask = stance_input_mask.to(device)
                stance_label_ids = stance_label_ids.to(device)
                #stance_stance_position = stance_stance_position.to(device)
                stance_label_mask = stance_label_mask.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_mask, label_ids, stance_label_mask=label_mask)
                    logits, _ = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_mask, stance_label_mask=label_mask)
                    stance_logits = model(stance_input_ids1, stance_segment_ids1, stance_input_mask1, stance_input_ids2,
                                    stance_segment_ids2, stance_input_mask2, stance_input_ids3, stance_segment_ids3,
                                    stance_input_mask3, stance_input_ids4, stance_segment_ids4, stance_input_mask4,
                                    stance_input_mask, task='stance', stance_label_mask=stance_label_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                stance_logits = torch.argmax(F.log_softmax(stance_logits, dim=2), dim=2)
                stance_logits = stance_logits.detach().cpu().numpy()
                stance_label_ids = stance_label_ids.to('cpu').numpy()
                stance_label_mask = stance_label_mask.to('cpu').numpy()
                for i, mask in enumerate(stance_label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            temp_1.append(stance_label_map[stance_label_ids[i][j]])
                            temp_2.append(stance_label_map[stance_logits[i][j]])
                        else:
                            break
                    stance_y_true.append(temp_1)
                    stance_y_pred.append(temp_2)

                nb_eval_examples += input_ids1.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = rumor_macro_f1(true_label, pred_outputs)

            stance_report = classification_report(stance_y_true, stance_y_pred, digits=4)
            logger.info("\n***** Test Stance Eval results *****")
            logger.info("\n%s", stance_report)
            stance_eval_true_label = np.concatenate(stance_y_true)
            stance_eval_pred_label = np.concatenate(stance_y_pred)
            stance_precision, stance_recall, stance_F_score = macro_f1(stance_eval_true_label, stance_eval_pred_label)
            print("stance_F-score: ", stance_F_score)

            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'f_score': F_score,
                      'global_step': global_step,
                      'loss': loss}

            logger.info("\n***** Test Rumor Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

    # Load a trained model that you have fine-tuned

    model_state_dict = torch.load(output_model_file)
    if args.mt_model == 'FS':
        model = FullySharedBert.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                rumor_num_labels=num_labels, stance_num_labels=stance_num_labels,
                                                max_tweet_num=args.max_tweet_num, max_tweet_length=args.max_tweet_length,
                                                convert_size=args.convert_size)
    elif args.mt_model == 'DB':
        model = DualBert.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                         rumor_num_labels=num_labels, stance_num_labels=stance_num_labels,
                                         max_tweet_num=args.max_tweet_num, max_tweet_length=args.max_tweet_length,
                                         convert_size=args.convert_size)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        stance_eval_examples = stance_processor.get_test_examples(args.data_dir2)
        tweets_list = []
        stances_list = []
        for (ex_index, example) in enumerate(eval_examples):
            tweets_list.append(example.text_a)
        for (ex_index, example) in enumerate(stance_eval_examples):
            stances_list.append(example.label)

        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_tweet_num, args.max_tweet_length)
        logger.info("\n***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids1 = torch.tensor([f.input_ids1 for f in eval_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask1 for f in eval_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids1 for f in eval_features], dtype=torch.long)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in eval_features], dtype=torch.long)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in eval_features], dtype=torch.long)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in eval_features], dtype=torch.long)
        all_input_ids3 = torch.tensor([f.input_ids3 for f in eval_features], dtype=torch.long)
        all_input_mask3 = torch.tensor([f.input_mask3 for f in eval_features], dtype=torch.long)
        all_segment_ids3 = torch.tensor([f.segment_ids3 for f in eval_features], dtype=torch.long)
        all_input_ids4 = torch.tensor([f.input_ids4 for f in eval_features], dtype=torch.long)
        all_input_mask4 = torch.tensor([f.input_mask4 for f in eval_features], dtype=torch.long)
        all_segment_ids4 = torch.tensor([f.segment_ids4 for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                  all_input_ids2, all_input_mask2, all_segment_ids2,
                                  all_input_ids3, all_input_mask3, all_segment_ids3,
                                  all_input_ids4, all_input_mask4, all_segment_ids4,
                                  all_input_mask, all_label_ids, all_label_mask)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        true_label_list = []
        pred_label_list = []
        attention_probs_list = []
        stance_pred_label_list = []
        stance_label_map = {i: label for i, label in enumerate(stance_label_list, 1)}
        stance_label_map[0] = "PAD"
        #convert_stance_label_map = {-1: -1, 0: 0, 1: 2, 2: 1, 3: 3}  # this will solve the problem above
 
        for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_mask, label_ids, label_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids1 = input_ids1.to(device)
            input_mask1 = input_mask1.to(device)
            segment_ids1 = segment_ids1.to(device)
            input_ids2 = input_ids2.to(device)
            input_mask2 = input_mask2.to(device)
            segment_ids2 = segment_ids2.to(device)
            input_ids3 = input_ids3.to(device)
            input_mask3 = input_mask3.to(device)
            segment_ids3 = segment_ids3.to(device)
            input_ids4 = input_ids4.to(device)
            input_mask4 = input_mask4.to(device)
            segment_ids4 = segment_ids4.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)
            label_mask = label_mask.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                      input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                      input_mask, label_ids, stance_label_mask=label_mask)
                logits, attention_probs = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_mask, stance_label_mask=label_mask)
                stance_logits = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_mask, task='stance', stance_label_mask=label_mask)

            logits = logits.detach().cpu().numpy()
            if args.mt_model == 'DB':
                attention_probs = attention_probs.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            if args.mt_model == 'DB':
                stance_logits = torch.argmax(F.log_softmax(stance_logits, dim=2), dim=2)
                stance_logits = stance_logits.detach().cpu().numpy()
                stance_label_mask = label_mask.to('cpu').numpy()
                for i, mask in enumerate(stance_label_mask):
                    temp_1 = []
                    attention_probs_temp_1 = []
                    for j, m in enumerate(mask):
                        if m:
                            temp_1.append(str(stance_logits[i][j] - 1))
                            attention_probs_temp_1.append(str(attention_probs[i][j]))
                            # 4 to 3, 3 to 2, 2 to 1, 1 to 0, same as original
                        else:
                            break
                    stance_pred_label_list.append(temp_1)
                    attention_probs_list.append(attention_probs_temp_1)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids1.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)
        #stance_pred_outputs_numpy = np.concatenate(stance_pred_label_list)

        precision, recall, F_score = rumor_macro_f1(true_label, pred_outputs)
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'precision': precision,
                  'recall': recall,
                  'f_score': F_score,
                  'global_step': global_step,
                  'loss': loss}

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
        fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + '\n')
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + '\n')

        fout_p.close()
        fout_t.close()
        FR_stance_dict = {'0':0, '1':0, '2':0, '3':0}  # deny, support, query, comment
        TR_stance_dict = {'0':0, '1':0, '2':0, '3':0}  # deny, support, query, comment
        UR_stance_dict = {'0':0, '1':0, '2':0, '3':0}  # deny, support, query, comment
        if args.task_name.lower() == "semeval17" and args.mt_model == 'DB':
            fout_analysis = open(os.path.join(args.output_dir, "analysis.txt"), 'w')
            for i in range(len(pred_label)):
                fout_analysis.write('Test Sample: ' + str(i) + '\n')
                tweets = tweets_list[i]
                fout_analysis.write('|||||'.join(tweets) + '\n')
                pred_probs = attention_probs_list[i]
                fout_analysis.write('attention stance probs: ' + ','.join(pred_probs) + '\n')
                pred_stances = stance_pred_label_list[i]
                fout_analysis.write('predicted stance label: ' + ','.join(pred_stances) + '\n')
                stances = stances_list[i]
                fout_analysis.write('original stance label:  ' + ','.join(stances) + '\n')
                attstr = str(pred_label[i])
                fout_analysis.write('predicted rumor label: ' + attstr + '\n')
                attstr = str(true_label[i])
                fout_analysis.write('true rumor label: ' + attstr + '\n\n')
                if pred_label[i] == 0:
                    for stance in pred_stances:
                        FR_stance_dict[stance] += 1
                elif pred_label[i] == 1:
                    for stance in pred_stances:
                        TR_stance_dict[stance] += 1
                elif pred_label[i] == 2:
                    for stance in pred_stances:
                        UR_stance_dict[stance] += 1
            fr_total = float(FR_stance_dict['0']+FR_stance_dict['1']+FR_stance_dict['2'])
            tr_total = float(TR_stance_dict['0']+TR_stance_dict['1']+TR_stance_dict['2'])
            ur_total = float(UR_stance_dict['0']+UR_stance_dict['1']+UR_stance_dict['2'])
            fout_analysis.write( '\n\n\nFalse Rumor------'+'deny: ' + str(FR_stance_dict['0']/fr_total)+'\t'+'support: ' +
                                str(FR_stance_dict['1']/fr_total)+'\t'+'query: ' +
                                str(FR_stance_dict['2']/fr_total) + '\n')
            fout_analysis.write( 'True Rumor------'+'deny: ' + str(TR_stance_dict['0']/tr_total)+'\t'+'support: ' +
                                str(TR_stance_dict['1']/tr_total)+'\t'+'query: ' +
                                str(TR_stance_dict['2']/tr_total) + '\n')
            fout_analysis.write( 'Unverified Rumor------'+'deny: ' + str(UR_stance_dict['0']/ur_total)+'\t'+'support: ' +
                                str(UR_stance_dict['1']/ur_total)+'\t'+'query: ' +
                                str(UR_stance_dict['2']/ur_total) + '\n')

            fout_analysis.close()
        elif args.task_name.lower() == "pheme" and args.mt_model == 'DB':
            fout_analysis = open(os.path.join(args.output_dir, "analysis.txt"), 'w')
            for i in range(len(pred_label)):
                fout_analysis.write('Test Sample: ' + str(i) + '\n')
                tweets = tweets_list[i]
                fout_analysis.write('|||||'.join(tweets) + '\n')
                pred_probs = attention_probs_list[i]
                fout_analysis.write('attention stance probs: ' + ','.join(pred_probs) + '\n')
                pred_stances = stance_pred_label_list[i]
                fout_analysis.write('predicted stance label: ' + ','.join(pred_stances) + '\n')
                attstr = str(pred_label[i])
                fout_analysis.write('predicted rumor label: ' + attstr + '\n')
                attstr = str(true_label[i])
                fout_analysis.write('true rumor label: ' + attstr + '\n\n')
                if pred_label[i] == 0:
                    for stance in pred_stances:
                        FR_stance_dict[stance] += 1
                elif pred_label[i] == 1:
                    for stance in pred_stances:
                        TR_stance_dict[stance] += 1
                elif pred_label[i] == 2:
                    for stance in pred_stances:
                        UR_stance_dict[stance] += 1

            fr_total = float(FR_stance_dict['0']+FR_stance_dict['1']+FR_stance_dict['2'])
            tr_total = float(TR_stance_dict['0']+TR_stance_dict['1']+TR_stance_dict['2'])
            ur_total = float(UR_stance_dict['0']+UR_stance_dict['1']+UR_stance_dict['2'])
            fout_analysis.write( '\n\n\nFalse Rumor------'+'deny: ' + str(FR_stance_dict['0']/fr_total)+'\t'+'support: ' +
                                str(FR_stance_dict['1']/fr_total)+'\t'+'query: ' +
                                str(FR_stance_dict['2']/fr_total) + '\n')
            fout_analysis.write( 'True Rumor------'+'deny: ' + str(TR_stance_dict['0']/tr_total)+'\t'+'support: ' +
                                str(TR_stance_dict['1']/tr_total)+'\t'+'query: ' +
                                str(TR_stance_dict['2']/tr_total) + '\n')
            fout_analysis.write( 'Unverified Rumor------'+'deny: ' + str(UR_stance_dict['0']/ur_total)+'\t'+'support: ' +
                                str(UR_stance_dict['1']/ur_total)+'\t'+'query: ' +
                                str(UR_stance_dict['2']/ur_total) + '\n')

            fout_analysis.close()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("\n***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
