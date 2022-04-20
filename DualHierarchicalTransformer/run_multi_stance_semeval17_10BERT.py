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
from my_bert.modeling_10BERT import BertForSeqStanceClassification
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputStanceFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2,
                 input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4,
                 input_ids5, input_mask5, segment_ids5, input_ids6, input_mask6, segment_ids6,
                 input_ids7, input_mask7, segment_ids7, input_ids8, input_mask8, segment_ids8,
                 input_ids9, input_mask9, segment_ids9, input_ids10, input_mask10, segment_ids10, input_mask,
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
        self.input_ids5 = input_ids5
        self.input_mask5 = input_mask5
        self.segment_ids5 = segment_ids5
        self.input_ids6 = input_ids6
        self.input_mask6 = input_mask6
        self.segment_ids6 = segment_ids6
        self.input_ids7 = input_ids7
        self.input_mask7 = input_mask7
        self.segment_ids7 = segment_ids7
        self.input_ids8 = input_ids8
        self.input_mask8 = input_mask8
        self.segment_ids8 = segment_ids8
        self.input_ids9 = input_ids9
        self.input_mask9 = input_mask9
        self.segment_ids9 = segment_ids9
        self.input_ids10 = input_ids10
        self.input_mask10 = input_mask10
        self.segment_ids10 = segment_ids10
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
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2].lower()
            text_b = line[3].lower()
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

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


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

    # max_tweet_len = 34  # the number of words in each tweet
    # max_tweet_num = 15  # the number of tweets in each bucket
    max_bucket_num = 10  # the number of buckets in each thread
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
            tweets_tokens5, labels5, tweets_tokens6, labels6, tweets_tokens7, labels7 = [], [], [], [], [], []
            tweets_tokens8, labels8, tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], [], [], []
        elif len(labels) > max_tweet_num and len(labels) <= max_tweet_num*2:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:]
            labels2 = labels[max_tweet_num:]
            tweets_tokens3, labels3, tweets_tokens4, labels4 = [], [], [], []
            tweets_tokens5, labels5, tweets_tokens6, labels6, tweets_tokens7, labels7 = [], [], [], [], [], []
            tweets_tokens8, labels8, tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], [], [], []
        elif len(labels) > max_tweet_num*2 and len(labels) <= max_tweet_num*3:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:]
            labels3 = labels[max_tweet_num*2:]
            tweets_tokens4, labels4 = [], []
            tweets_tokens5, labels5, tweets_tokens6, labels6, tweets_tokens7, labels7 = [], [], [], [], [], []
            tweets_tokens8, labels8, tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], [], [], []
        elif len(labels) > max_tweet_num*3 and len(labels) <= max_tweet_num*4:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:]
            labels4 = labels[max_tweet_num*3:]
            tweets_tokens5, labels5, tweets_tokens6, labels6, tweets_tokens7, labels7 = [], [], [], [], [], []
            tweets_tokens8, labels8, tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], [], [], []
        elif len(labels) > max_tweet_num*4 and len(labels) <= max_tweet_num*5:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]
            tweets_tokens5 = tweets_tokens[max_tweet_num*4:]
            labels5 = labels[max_tweet_num*4:]
            tweets_tokens6, labels6, tweets_tokens7, labels7, tweets_tokens8, labels8 = [], [], [], [], [], []
            tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], []
        elif len(labels) > max_tweet_num*5 and len(labels) <= max_tweet_num*6:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]
            tweets_tokens5 = tweets_tokens[max_tweet_num*4:max_tweet_num*5]
            labels5 = labels[max_tweet_num*4:max_tweet_num*5]
            tweets_tokens6 = tweets_tokens[max_tweet_num*5:]
            labels6 = labels[max_tweet_num*5:]
            tweets_tokens7, labels7, tweets_tokens8, labels8 = [], [], [], []
            tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], []
        elif len(labels) > max_tweet_num*6 and len(labels) <= max_tweet_num*7:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]
            tweets_tokens5 = tweets_tokens[max_tweet_num*4:max_tweet_num*5]
            labels5 = labels[max_tweet_num*4:max_tweet_num*5]
            tweets_tokens6 = tweets_tokens[max_tweet_num*5:max_tweet_num*6]
            labels6 = labels[max_tweet_num*5:max_tweet_num*6]
            tweets_tokens7 = tweets_tokens[max_tweet_num*6:]
            labels7 = labels[max_tweet_num*6:]
            tweets_tokens8, labels8 = [], []
            tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], []
        elif len(labels) > max_tweet_num*7 and len(labels) <= max_tweet_num*8:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]
            tweets_tokens5 = tweets_tokens[max_tweet_num*4:max_tweet_num*5]
            labels5 = labels[max_tweet_num*4:max_tweet_num*5]
            tweets_tokens6 = tweets_tokens[max_tweet_num*5:max_tweet_num*6]
            labels6 = labels[max_tweet_num*5:max_tweet_num*6]
            tweets_tokens7 = tweets_tokens[max_tweet_num*6:max_tweet_num*7]
            labels7 = labels[max_tweet_num*6:max_tweet_num*7]
            tweets_tokens8 = tweets_tokens[max_tweet_num*7:]
            labels8 = labels[max_tweet_num*7:]
            tweets_tokens9, labels9, tweets_tokens10, labels10 = [], [], [], []
        elif len(labels) > max_tweet_num*8 and len(labels) <= max_tweet_num*9:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]
            tweets_tokens5 = tweets_tokens[max_tweet_num*4:max_tweet_num*5]
            labels5 = labels[max_tweet_num*4:max_tweet_num*5]
            tweets_tokens6 = tweets_tokens[max_tweet_num*5:max_tweet_num*6]
            labels6 = labels[max_tweet_num*5:max_tweet_num*6]
            tweets_tokens7 = tweets_tokens[max_tweet_num*6:max_tweet_num*7]
            labels7 = labels[max_tweet_num*6:max_tweet_num*7]
            tweets_tokens8 = tweets_tokens[max_tweet_num*7:max_tweet_num*8]
            labels8 = labels[max_tweet_num*7:max_tweet_num*8]
            tweets_tokens9 = tweets_tokens[max_tweet_num*8:]
            labels9 = labels[max_tweet_num*8:]
            tweets_tokens10, labels10 = [], []
        else:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            labels1 = labels[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:max_tweet_num*2]
            labels2 = labels[max_tweet_num:max_tweet_num*2]
            tweets_tokens3 = tweets_tokens[max_tweet_num*2:max_tweet_num*3]
            labels3 = labels[max_tweet_num*2:max_tweet_num*3]
            tweets_tokens4 = tweets_tokens[max_tweet_num*3:max_tweet_num*4]
            labels4 = labels[max_tweet_num*3:max_tweet_num*4]
            tweets_tokens5 = tweets_tokens[max_tweet_num*4:max_tweet_num*5]
            labels5 = labels[max_tweet_num*4:max_tweet_num*5]
            tweets_tokens6 = tweets_tokens[max_tweet_num*5:max_tweet_num*6]
            labels6 = labels[max_tweet_num*5:max_tweet_num*6]
            tweets_tokens7 = tweets_tokens[max_tweet_num*6:max_tweet_num*7]
            labels7 = labels[max_tweet_num*6:max_tweet_num*7]
            tweets_tokens8 = tweets_tokens[max_tweet_num*7:max_tweet_num*8]
            labels8 = labels[max_tweet_num*7:max_tweet_num*8]
            tweets_tokens9 = tweets_tokens[max_tweet_num*8:max_tweet_num*9]
            labels9 = labels[max_tweet_num*8:max_tweet_num*9]
            tweets_tokens10 = tweets_tokens[max_tweet_num*9:max_tweet_num*10]
            labels10 = labels[max_tweet_num*9:max_tweet_num*10]

        input_tokens1, input_ids1, input_mask1, segment_ids1, label_ids1, stance_position1, label_mask1 = \
            bucket_conversion(tweets_tokens1, labels1, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens2, input_ids2, input_mask2, segment_ids2, label_ids2, stance_position2, label_mask2 = \
            bucket_conversion(tweets_tokens2, labels2, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens3, input_ids3, input_mask3, segment_ids3, label_ids3, stance_position3, label_mask3 = \
            bucket_conversion(tweets_tokens3, labels3, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens4, input_ids4, input_mask4, segment_ids4, label_ids4, stance_position4, label_mask4 = \
            bucket_conversion(tweets_tokens4, labels4, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens5, input_ids5, input_mask5, segment_ids5, label_ids5, stance_position5, label_mask5 = \
            bucket_conversion(tweets_tokens5, labels5, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens6, input_ids6, input_mask6, segment_ids6, label_ids6, stance_position6, label_mask6 = \
            bucket_conversion(tweets_tokens6, labels6, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens7, input_ids7, input_mask7, segment_ids7, label_ids7, stance_position7, label_mask7 = \
            bucket_conversion(tweets_tokens7, labels7, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens8, input_ids8, input_mask8, segment_ids8, label_ids8, stance_position8, label_mask8 = \
            bucket_conversion(tweets_tokens8, labels8, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens9, input_ids9, input_mask9, segment_ids9, label_ids9, stance_position9, label_mask9 = \
            bucket_conversion(tweets_tokens9, labels9, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)
        input_tokens10, input_ids10, input_mask10, segment_ids10, label_ids10, stance_position10, label_mask10 = \
            bucket_conversion(tweets_tokens10, labels10, label_map, tokenizer, max_tweet_num, max_tweet_len, max_seq_length)

        label_ids = []
        label_ids.extend(label_ids1)
        label_ids.extend(label_ids2)
        label_ids.extend(label_ids3)
        label_ids.extend(label_ids4)
        label_ids.extend(label_ids5)
        label_ids.extend(label_ids6)
        label_ids.extend(label_ids7)
        label_ids.extend(label_ids8)
        label_ids.extend(label_ids9)
        label_ids.extend(label_ids10)
        stance_position =[]
        stance_position.extend(stance_position1)
        stance_position.extend(stance_position2)
        stance_position.extend(stance_position3)
        stance_position.extend(stance_position4)
        stance_position.extend(stance_position5)
        stance_position.extend(stance_position6)
        stance_position.extend(stance_position7)
        stance_position.extend(stance_position8)
        stance_position.extend(stance_position9)
        stance_position.extend(stance_position10)
        label_mask = []
        label_mask.extend(label_mask1)
        label_mask.extend(label_mask2)
        label_mask.extend(label_mask3)
        label_mask.extend(label_mask4)
        label_mask.extend(label_mask5)
        label_mask.extend(label_mask6)
        label_mask.extend(label_mask7)
        label_mask.extend(label_mask8)
        label_mask.extend(label_mask9)
        label_mask.extend(label_mask10)
        input_mask = []
        input_mask.extend(input_mask1)
        input_mask.extend(input_mask2)
        input_mask.extend(input_mask3)
        input_mask.extend(input_mask4)
        input_mask.extend(input_mask5)
        input_mask.extend(input_mask6)
        input_mask.extend(input_mask7)
        input_mask.extend(input_mask8)
        input_mask.extend(input_mask9)
        input_mask.extend(input_mask10)

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
                                input_ids5=input_ids5, input_mask5=input_mask5, segment_ids5=segment_ids5,
                                input_ids6=input_ids6, input_mask6=input_mask6, segment_ids6=segment_ids6,
                                input_ids7=input_ids7, input_mask7=input_mask7, segment_ids7=segment_ids7,
                                input_ids8=input_ids8, input_mask8=input_mask8, segment_ids8=segment_ids8,
                                input_ids9=input_ids9, input_mask9=input_mask9, segment_ids9=segment_ids9,
                                input_ids10=input_ids10, input_mask10=input_mask10, segment_ids10=segment_ids10,
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

def stance_accuracy(out, labels):
    correct = np.sum(out == labels) # np.argmax(out, axis=1)
    acc = float(correct)/len(labels)
    return acc

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
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='twitter',
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
                        default=2,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=25.0,
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
                        default=64,
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
    parser.add_argument('--max_tweet_num', type=int, default=30, help="the maximum number of tweets")
    parser.add_argument('--max_tweet_length', type=int, default=17, help="the maximum length of each tweet")


    args = parser.parse_args()

    if args.bertlayer:
        print("add another bert layer")
    else:
        print("pre-trained bert without additional bert layer")

    #if args.task_name == "rumor2015" and args.bertlayer:
        #args.seed += 22

    processors = {
        "semeval17_stance": StanceProcessor
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

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1 # label 0 corresponds to padding, label in label_list starts from 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSeqStanceClassification.from_pretrained(args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
              num_labels = num_labels, max_tweet_num=args.max_tweet_num, max_tweet_length=args.max_tweet_length)
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
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        print('training data')
        train_features = convert_stance_examples_to_features(
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
        all_input_ids5 = torch.tensor([f.input_ids5 for f in train_features], dtype=torch.long)
        all_input_mask5 = torch.tensor([f.input_mask5 for f in train_features], dtype=torch.long)
        all_segment_ids5 = torch.tensor([f.segment_ids5 for f in train_features], dtype=torch.long)
        all_input_ids6 = torch.tensor([f.input_ids6 for f in train_features], dtype=torch.long)
        all_input_mask6 = torch.tensor([f.input_mask6 for f in train_features], dtype=torch.long)
        all_segment_ids6 = torch.tensor([f.segment_ids6 for f in train_features], dtype=torch.long)
        all_input_ids7 = torch.tensor([f.input_ids7 for f in train_features], dtype=torch.long)
        all_input_mask7 = torch.tensor([f.input_mask7 for f in train_features], dtype=torch.long)
        all_segment_ids7 = torch.tensor([f.segment_ids7 for f in train_features], dtype=torch.long)
        all_input_ids8 = torch.tensor([f.input_ids8 for f in train_features], dtype=torch.long)
        all_input_mask8 = torch.tensor([f.input_mask8 for f in train_features], dtype=torch.long)
        all_segment_ids8 = torch.tensor([f.segment_ids8 for f in train_features], dtype=torch.long)
        all_input_ids9 = torch.tensor([f.input_ids9 for f in train_features], dtype=torch.long)
        all_input_mask9 = torch.tensor([f.input_mask9 for f in train_features], dtype=torch.long)
        all_segment_ids9 = torch.tensor([f.segment_ids9 for f in train_features], dtype=torch.long)
        all_input_ids10 = torch.tensor([f.input_ids10 for f in train_features], dtype=torch.long)
        all_input_mask10 = torch.tensor([f.input_mask10 for f in train_features], dtype=torch.long)
        all_segment_ids10 = torch.tensor([f.segment_ids10 for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        #all_stance_position = torch.tensor([f.stance_position for f in train_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                   all_input_ids2, all_input_mask2, all_segment_ids2,
                                   all_input_ids3, all_input_mask3, all_segment_ids3,
                                   all_input_ids4, all_input_mask4, all_segment_ids4,
                                   all_input_ids5, all_input_mask5, all_segment_ids5,
                                   all_input_ids6, all_input_mask6, all_segment_ids6,
                                   all_input_ids7, all_input_mask7, all_segment_ids7,
                                   all_input_ids8, all_input_mask8, all_segment_ids8,
                                   all_input_ids9, all_input_mask9, all_segment_ids9,
                                   all_input_ids10, all_input_mask10, all_segment_ids10,
                                   all_input_mask, all_label_ids, all_label_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        #'''
        eval_examples = processor.get_dev_examples(args.data_dir)
        print('dev data')
        eval_features = convert_stance_examples_to_features(
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
        all_input_ids5 = torch.tensor([f.input_ids5 for f in eval_features], dtype=torch.long)
        all_input_mask5 = torch.tensor([f.input_mask5 for f in eval_features], dtype=torch.long)
        all_segment_ids5 = torch.tensor([f.segment_ids5 for f in eval_features], dtype=torch.long)
        all_input_ids6 = torch.tensor([f.input_ids6 for f in eval_features], dtype=torch.long)
        all_input_mask6 = torch.tensor([f.input_mask6 for f in eval_features], dtype=torch.long)
        all_segment_ids6 = torch.tensor([f.segment_ids6 for f in eval_features], dtype=torch.long)
        all_input_ids7 = torch.tensor([f.input_ids7 for f in eval_features], dtype=torch.long)
        all_input_mask7 = torch.tensor([f.input_mask7 for f in eval_features], dtype=torch.long)
        all_segment_ids7 = torch.tensor([f.segment_ids7 for f in eval_features], dtype=torch.long)
        all_input_ids8 = torch.tensor([f.input_ids8 for f in eval_features], dtype=torch.long)
        all_input_mask8 = torch.tensor([f.input_mask8 for f in eval_features], dtype=torch.long)
        all_segment_ids8 = torch.tensor([f.segment_ids8 for f in eval_features], dtype=torch.long)
        all_input_ids9 = torch.tensor([f.input_ids9 for f in eval_features], dtype=torch.long)
        all_input_mask9 = torch.tensor([f.input_mask9 for f in eval_features], dtype=torch.long)
        all_segment_ids9 = torch.tensor([f.segment_ids9 for f in eval_features], dtype=torch.long)
        all_input_ids10 = torch.tensor([f.input_ids10 for f in eval_features], dtype=torch.long)
        all_input_mask10 = torch.tensor([f.input_mask10 for f in eval_features], dtype=torch.long)
        all_segment_ids10 = torch.tensor([f.segment_ids10 for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        #all_stance_position = torch.tensor([f.stance_position for f in eval_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                  all_input_ids2, all_input_mask2, all_segment_ids2,
                                  all_input_ids3, all_input_mask3, all_segment_ids3,
                                  all_input_ids4, all_input_mask4, all_segment_ids4,
                                  all_input_ids5, all_input_mask5, all_segment_ids5,
                                  all_input_ids6, all_input_mask6, all_segment_ids6,
                                  all_input_ids7, all_input_mask7, all_segment_ids7,
                                  all_input_ids8, all_input_mask8, all_segment_ids8,
                                  all_input_ids9, all_input_mask9, all_segment_ids9,
                                  all_input_ids10, all_input_mask10, all_segment_ids10,
                                  all_input_mask, all_label_ids, all_label_mask)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # test_dataloader
        test_examples = processor.get_test_examples(args.data_dir)
        print('test data')
        test_features = convert_stance_examples_to_features(
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
        all_input_ids5 = torch.tensor([f.input_ids5 for f in test_features], dtype=torch.long)
        all_input_mask5 = torch.tensor([f.input_mask5 for f in test_features], dtype=torch.long)
        all_segment_ids5 = torch.tensor([f.segment_ids5 for f in test_features], dtype=torch.long)
        all_input_ids6 = torch.tensor([f.input_ids6 for f in test_features], dtype=torch.long)
        all_input_mask6 = torch.tensor([f.input_mask6 for f in test_features], dtype=torch.long)
        all_segment_ids6 = torch.tensor([f.segment_ids6 for f in test_features], dtype=torch.long)
        all_input_ids7 = torch.tensor([f.input_ids7 for f in test_features], dtype=torch.long)
        all_input_mask7 = torch.tensor([f.input_mask7 for f in test_features], dtype=torch.long)
        all_segment_ids7 = torch.tensor([f.segment_ids7 for f in test_features], dtype=torch.long)
        all_input_ids8 = torch.tensor([f.input_ids8 for f in test_features], dtype=torch.long)
        all_input_mask8 = torch.tensor([f.input_mask8 for f in test_features], dtype=torch.long)
        all_segment_ids8 = torch.tensor([f.segment_ids8 for f in test_features], dtype=torch.long)
        all_input_ids9 = torch.tensor([f.input_ids9 for f in test_features], dtype=torch.long)
        all_input_mask9 = torch.tensor([f.input_mask9 for f in test_features], dtype=torch.long)
        all_segment_ids9 = torch.tensor([f.segment_ids9 for f in test_features], dtype=torch.long)
        all_input_ids10 = torch.tensor([f.input_ids10 for f in test_features], dtype=torch.long)
        all_input_mask10 = torch.tensor([f.input_mask10 for f in test_features], dtype=torch.long)
        all_segment_ids10 = torch.tensor([f.segment_ids10 for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        #all_stance_position = torch.tensor([f.stance_position for f in eval_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                  all_input_ids2, all_input_mask2, all_segment_ids2,
                                  all_input_ids3, all_input_mask3, all_segment_ids3,
                                  all_input_ids4, all_input_mask4, all_segment_ids4,
                                  all_input_ids5, all_input_mask5, all_segment_ids5,
                                  all_input_ids6, all_input_mask6, all_segment_ids6,
                                  all_input_ids7, all_input_mask7, all_segment_ids7,
                                  all_input_ids8, all_input_mask8, all_segment_ids8,
                                  all_input_ids9, all_input_mask9, all_segment_ids9,
                                  all_input_ids10, all_input_mask10, all_segment_ids10,
                                  all_input_mask, all_label_ids, all_label_mask)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        max_f1 = 0.0
        #'''
        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):

            logger.info("********** Epoch: "+ str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_ids5, input_mask5, segment_ids5, input_ids6, input_mask6, segment_ids6, \
                input_ids7, input_mask7, segment_ids7, input_ids8, input_mask8, segment_ids8, \
                input_ids9, input_mask9, segment_ids9, input_ids10, input_mask10, segment_ids10, \
                input_mask, label_ids, label_mask = batch
                loss = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                             input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                             input_ids5, segment_ids5, input_mask5, input_ids6, segment_ids6, input_mask6,
                             input_ids7, segment_ids7, input_mask7, input_ids8, segment_ids8, input_mask8,
                             input_ids9, segment_ids9, input_mask9, input_ids10, segment_ids10, input_mask10,
                             input_mask, label_ids, label_mask)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
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
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            '''
            logger.info("***** Running evaluation on Train Set*****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0

            y_true = []
            y_pred = []

            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_ids5, input_mask5, segment_ids5, input_ids6, input_mask6, segment_ids6, \
                input_ids7, input_mask7, segment_ids7, input_ids8, input_mask8, segment_ids8, \
                input_ids9, input_mask9, segment_ids9, input_ids10, input_mask10, segment_ids10, \
                input_mask, label_ids, label_mask in tqdm(train_dataloader, desc="Evaluating"):
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
                input_ids5 = input_ids5.to(device)
                input_mask5 = input_mask5.to(device)
                segment_ids5 = segment_ids5.to(device)
                input_ids6 = input_ids6.to(device)
                input_mask6 = input_mask6.to(device)
                segment_ids6 = segment_ids6.to(device)
                input_ids7 = input_ids7.to(device)
                input_mask7 = input_mask7.to(device)
                segment_ids7 = segment_ids7.to(device)
                input_ids8 = input_ids8.to(device)
                input_mask8 = input_mask8.to(device)
                segment_ids8 = segment_ids8.to(device)
                input_ids9 = input_ids9.to(device)
                input_mask9 = input_mask9.to(device)
                segment_ids9 = segment_ids9.to(device)
                input_ids10 = input_ids10.to(device)
                input_mask10 = input_mask10.to(device)
                segment_ids10 = segment_ids10.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)
                #stance_position = stance_position.to(device)
                label_mask = label_mask.to(device)

                with torch.no_grad():
                    #tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, label_mask)
                    logits = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_ids5, segment_ids5, input_mask5, input_ids6, segment_ids6, input_mask6,
                                   input_ids7, segment_ids7, input_mask7, input_ids8, segment_ids8, input_mask8,
                                   input_ids9, segment_ids9, input_mask9, input_ids10, segment_ids10, input_mask10,
                                   input_mask, label_mask=label_mask)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                label_mask = label_mask.to('cpu').numpy()
                for i, mask in enumerate(label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            if label_map[label_ids[i][j]] != "[SEP]" and label_map[label_ids[i][j]] != "[CLS]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Train Eval results *****")
            logger.info("\n%s", report)
            eval_true_label = np.concatenate(y_true)
            eval_pred_label = np.concatenate(y_pred)
            precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
            print("F-score: ", F_score)
            '''

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0

            y_true = []
            y_pred = []

            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_ids5, input_mask5, segment_ids5, input_ids6, input_mask6, segment_ids6, \
                input_ids7, input_mask7, segment_ids7, input_ids8, input_mask8, segment_ids8, \
                input_ids9, input_mask9, segment_ids9, input_ids10, input_mask10, segment_ids10, \
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
                input_ids5 = input_ids5.to(device)
                input_mask5 = input_mask5.to(device)
                segment_ids5 = segment_ids5.to(device)
                input_ids6 = input_ids6.to(device)
                input_mask6 = input_mask6.to(device)
                segment_ids6 = segment_ids6.to(device)
                input_ids7 = input_ids7.to(device)
                input_mask7 = input_mask7.to(device)
                segment_ids7 = segment_ids7.to(device)
                input_ids8 = input_ids8.to(device)
                input_mask8 = input_mask8.to(device)
                segment_ids8 = segment_ids8.to(device)
                input_ids9 = input_ids9.to(device)
                input_mask9 = input_mask9.to(device)
                segment_ids9 = segment_ids9.to(device)
                input_ids10 = input_ids10.to(device)
                input_mask10 = input_mask10.to(device)
                segment_ids10 = segment_ids10.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)
                #stance_position = stance_position.to(device)
                label_mask = label_mask.to(device)

                with torch.no_grad():
                    #tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, label_mask)
                    logits = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_ids5, segment_ids5, input_mask5, input_ids6, segment_ids6, input_mask6,
                                   input_ids7, segment_ids7, input_mask7, input_ids8, segment_ids8, input_mask8,
                                   input_ids9, segment_ids9, input_mask9, input_ids10, segment_ids10, input_mask10,
                                   input_mask, label_mask=label_mask)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                label_mask = label_mask.to('cpu').numpy()
                for i, mask in enumerate(label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            if label_map[label_ids[i][j]] != "[SEP]" and label_map[label_ids[i][j]] != "[CLS]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Dev Eval results *****")
            logger.info("\n%s", report)
            eval_true_label = np.concatenate(y_true)
            eval_pred_label = np.concatenate(y_pred)
            precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
            acc = stance_accuracy(eval_pred_label, eval_true_label)
            print("F-score: ", F_score)
            print("Accuracy: ", acc)

            if F_score>max_f1:
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                if args.do_train:
                    torch.save(model_to_save.state_dict(), output_model_file)
                max_f1 = F_score

            logger.info("***** Running evaluation on Test Set*****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0

            y_true = []
            y_pred = []

            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
                input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
                input_ids5, input_mask5, segment_ids5, input_ids6, input_mask6, segment_ids6, \
                input_ids7, input_mask7, segment_ids7, input_ids8, input_mask8, segment_ids8, \
                input_ids9, input_mask9, segment_ids9, input_ids10, input_mask10, segment_ids10, \
                input_mask, label_ids, label_mask in tqdm(test_dataloader, desc="Evaluating"):
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
                input_ids5 = input_ids5.to(device)
                input_mask5 = input_mask5.to(device)
                segment_ids5 = segment_ids5.to(device)
                input_ids6 = input_ids6.to(device)
                input_mask6 = input_mask6.to(device)
                segment_ids6 = segment_ids6.to(device)
                input_ids7 = input_ids7.to(device)
                input_mask7 = input_mask7.to(device)
                segment_ids7 = segment_ids7.to(device)
                input_ids8 = input_ids8.to(device)
                input_mask8 = input_mask8.to(device)
                segment_ids8 = segment_ids8.to(device)
                input_ids9 = input_ids9.to(device)
                input_mask9 = input_mask9.to(device)
                segment_ids9 = segment_ids9.to(device)
                input_ids10 = input_ids10.to(device)
                input_mask10 = input_mask10.to(device)
                segment_ids10 = segment_ids10.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)
                #stance_position = stance_position.to(device)
                label_mask = label_mask.to(device)

                with torch.no_grad():
                    #tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, label_mask)
                    logits = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                                   input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                                   input_ids5, segment_ids5, input_mask5, input_ids6, segment_ids6, input_mask6,
                                   input_ids7, segment_ids7, input_mask7, input_ids8, segment_ids8, input_mask8,
                                   input_ids9, segment_ids9, input_mask9, input_ids10, segment_ids10, input_mask10,
                                   input_mask, label_mask=label_mask)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                label_mask = label_mask.to('cpu').numpy()
                for i, mask in enumerate(label_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if m:
                            if label_map[label_ids[i][j]] != "[SEP]" and label_map[label_ids[i][j]] != "[CLS]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Test Eval results *****")
            logger.info("\n%s", report)
            eval_true_label = np.concatenate(y_true)
            eval_pred_label = np.concatenate(y_pred)
            precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
            acc = stance_accuracy(eval_pred_label, eval_true_label)
            print("F-score: ", F_score)
            print("Accuracy: ", acc)


    # Load a trained model that you have fine-tuned

    model_state_dict = torch.load(output_model_file)
    model = BertForSeqStanceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, \
                                                          num_labels=num_labels, max_tweet_num=args.max_tweet_num, max_tweet_length=args.max_tweet_length)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_stance_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_tweet_num, args.max_tweet_length)
        logger.info("***** Running evaluation on Test Set *****")
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
        all_input_ids5 = torch.tensor([f.input_ids5 for f in eval_features], dtype=torch.long)
        all_input_mask5 = torch.tensor([f.input_mask5 for f in eval_features], dtype=torch.long)
        all_segment_ids5 = torch.tensor([f.segment_ids5 for f in eval_features], dtype=torch.long)
        all_input_ids6 = torch.tensor([f.input_ids6 for f in eval_features], dtype=torch.long)
        all_input_mask6 = torch.tensor([f.input_mask6 for f in eval_features], dtype=torch.long)
        all_segment_ids6 = torch.tensor([f.segment_ids6 for f in eval_features], dtype=torch.long)
        all_input_ids7 = torch.tensor([f.input_ids7 for f in eval_features], dtype=torch.long)
        all_input_mask7 = torch.tensor([f.input_mask7 for f in eval_features], dtype=torch.long)
        all_segment_ids7 = torch.tensor([f.segment_ids7 for f in eval_features], dtype=torch.long)
        all_input_ids8 = torch.tensor([f.input_ids8 for f in eval_features], dtype=torch.long)
        all_input_mask8 = torch.tensor([f.input_mask8 for f in eval_features], dtype=torch.long)
        all_segment_ids8 = torch.tensor([f.segment_ids8 for f in eval_features], dtype=torch.long)
        all_input_ids9 = torch.tensor([f.input_ids9 for f in eval_features], dtype=torch.long)
        all_input_mask9 = torch.tensor([f.input_mask9 for f in eval_features], dtype=torch.long)
        all_segment_ids9 = torch.tensor([f.segment_ids9 for f in eval_features], dtype=torch.long)
        all_input_ids10 = torch.tensor([f.input_ids10 for f in eval_features], dtype=torch.long)
        all_input_mask10 = torch.tensor([f.input_mask10 for f in eval_features], dtype=torch.long)
        all_segment_ids10 = torch.tensor([f.segment_ids10 for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        #all_stance_position = torch.tensor([f.stance_position for f in eval_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                  all_input_ids2, all_input_mask2, all_segment_ids2,
                                  all_input_ids3, all_input_mask3, all_segment_ids3,
                                  all_input_ids4, all_input_mask4, all_segment_ids4,
                                  all_input_ids5, all_input_mask5, all_segment_ids5,
                                  all_input_ids6, all_input_mask6, all_segment_ids6,
                                  all_input_ids7, all_input_mask7, all_segment_ids7,
                                  all_input_ids8, all_input_mask8, all_segment_ids8,
                                  all_input_ids9, all_input_mask9, all_segment_ids9,
                                  all_input_ids10, all_input_mask10, all_segment_ids10,
                                  all_input_mask, all_label_ids, all_label_mask)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        y_true = []
        y_pred = []

        stance_pred_label_list = []
        stance_true_label_list = []

        label_map = {i: label for i, label in enumerate(label_list, 1)}
 
        for input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, \
            input_ids3, input_mask3, segment_ids3, input_ids4, input_mask4, segment_ids4, \
            input_ids5, input_mask5, segment_ids5, input_ids6, input_mask6, segment_ids6, \
            input_ids7, input_mask7, segment_ids7, input_ids8, input_mask8, segment_ids8, \
            input_ids9, input_mask9, segment_ids9, input_ids10, input_mask10, segment_ids10, \
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
            input_ids5 = input_ids5.to(device)
            input_mask5 = input_mask5.to(device)
            segment_ids5 = segment_ids5.to(device)
            input_ids6 = input_ids6.to(device)
            input_mask6 = input_mask6.to(device)
            segment_ids6 = segment_ids6.to(device)
            input_ids7 = input_ids7.to(device)
            input_mask7 = input_mask7.to(device)
            segment_ids7 = segment_ids7.to(device)
            input_ids8 = input_ids8.to(device)
            input_mask8 = input_mask8.to(device)
            segment_ids8 = segment_ids8.to(device)
            input_ids9 = input_ids9.to(device)
            input_mask9 = input_mask9.to(device)
            segment_ids9 = segment_ids9.to(device)
            input_ids10 = input_ids10.to(device)
            input_mask10 = input_mask10.to(device)
            segment_ids10 = segment_ids10.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)
            # stance_position = stance_position.to(device)
            label_mask = label_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids1, segment_ids1, input_mask1, input_ids2, segment_ids2, input_mask2,
                               input_ids3, segment_ids3, input_mask3, input_ids4, segment_ids4, input_mask4,
                               input_ids5, segment_ids5, input_mask5, input_ids6, segment_ids6, input_mask6,
                               input_ids7, segment_ids7, input_mask7, input_ids8, segment_ids8, input_mask8,
                               input_ids9, segment_ids9, input_mask9, input_ids10, segment_ids10, input_mask10,
                               input_mask, label_mask=label_mask)

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            label_mask = label_mask.to('cpu').numpy()
            for i, mask in enumerate(label_mask):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(mask):
                    if m:
                        if label_map[label_ids[i][j]] != "[SEP]" and label_map[label_ids[i][j]] != "[CLS]":
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[logits[i][j]])
                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("***** Test Eval results *****")
        logger.info("\n%s", report)
        eval_true_label = np.concatenate(y_true)
        eval_pred_label = np.concatenate(y_pred)
        precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
        acc = stance_accuracy(eval_pred_label, eval_true_label)
        print("F-score: ", F_score)
        print("Accuracy: ", acc)

        #pred_label = np.argmax(pred_outputs, axis=-1)
        #fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
        #fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')

        fout_analysis = open(os.path.join(args.output_dir, "analysis.txt"), 'w')
        for i in range(len(stance_pred_label_list)):
            fout_analysis.write('Test Sample: ' + str(i) + '\n')
            tweets = tweets_list[i]
            fout_analysis.write('|||||'.join(tweets) + '\n')
            pred_stances = stance_pred_label_list[i]
            fout_analysis.write('predicted stance label: ' + ','.join(pred_stances) + '\n')
            true_stances = stance_true_label_list[i]
            fout_analysis.write('true stance label:      ' + ','.join(true_stances) + '\n')
            stances = ori_stances_list[i]
            fout_analysis.write('original stance label:  ' + ','.join(stances) + '\n\n')
        fout_analysis.close()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            #logger.info("***** Test Eval results *****")
            #logger.info("\n%s", report)
            writer.write(report)

if __name__ == "__main__":
    main()
