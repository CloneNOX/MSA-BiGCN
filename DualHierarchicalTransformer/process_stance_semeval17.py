''' Script for downloading all GLUE data.

Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized, 
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).

mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
'''

import os
import sys
import shutil
import argparse
import json

def format_rumor(data_dir, fold_num):
    print("Processing...SemEval17...stance")
    rumor_dir = os.path.join(data_dir, 'split_'+str(fold_num))
    if not os.path.isdir(rumor_dir):
        os.mkdir(rumor_dir)

    rumor_train_file = os.path.join(rumor_dir, "stance_train.json")
    rumor_dev_file = os.path.join(rumor_dir, "stance_dev.json")
    rumor_test_file = os.path.join(rumor_dir, "stance_test.json")

    assert os.path.isfile(rumor_train_file), "Train data not found at %s" % rumor_train_file
    assert os.path.isfile(rumor_test_file), "Test data not found at %s" % rumor_test_file

    fin = open(rumor_train_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(rumor_dir, "stance_train.tsv"), 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 String\t#2 String\n")
        count = 0
        for i in range(len(lines)):
            count += 1
            input_dict = json.loads(lines[i])
            tweets = input_dict["tweets"]
            s1 = '|||||'.join(tweets)
            label_list = input_dict["label"]
            str_label_lst = [str(p) for p in label_list]
            label = ','.join(str_label_lst)
            train_fh.write("%s\t%s\t%s\n" % (count, label, s1))

    fin = open(rumor_dev_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(rumor_dir, "stance_dev.tsv"), 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 String\t#2 String\n")
        count = 0
        for i in range(len(lines)):
            count += 1
            input_dict = json.loads(lines[i])
            tweets = input_dict["tweets"]
            s1 = '|||||'.join(tweets)
            label_list = input_dict["label"]
            str_label_lst = [str(p) for p in label_list]
            label = ','.join(str_label_lst)
            train_fh.write("%s\t%s\t%s\n" % (count, label, s1))

    fin = open(rumor_test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(rumor_dir, "stance_test.tsv"), 'w') as test_fh:
        test_fh.write("index\t#1 Label\t#2 String\t#2 String\n")
        count = 0
        for i in range(len(lines)):
            count += 1
            input_dict = json.loads(lines[i])
            tweets = input_dict["tweets"]
            s1 = '|||||'.join(tweets)
            label_list = input_dict["label"]
            str_label_lst = [str(p) for p in label_list]
            label = ','.join(str_label_lst)
            test_fh.write("%s\t%s\t%s\n" % (count, label, s1))
    print("\tCompleted!")


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='./rumor_data/semeval17_stance/')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all') #all rumor2015
    parser.add_argument('--path_to_mrpc', help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    for i in range(1):
        format_rumor(args.data_dir, i)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))