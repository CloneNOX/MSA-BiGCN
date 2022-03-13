import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import argparse
import numpy as np
from data import *
from ABGCN import *
from sklearn.metrics import f1_score
from utils import *
from random import randint, shuffle
from tqdm import tqdm
from copy import copy
import sys

parser = argparse.ArgumentParser(description="Training script for ABGCN")

# model hyperparameters
parser.add_argument('--w2vDim', type=int, default=100,\
                    help='dimention of word2vec, default: 100')
parser.add_argument('--s2vDim' ,type=int, default=128,\
                    help='dimention of sentence2vec(get from lstm/attention)')
parser.add_argument('--gcnHiddenDim', type=int, default=128,\
                    help='dimention of GCN hidden layer, default: 128')
parser.add_argument('--rumorFeatureDim', type=int, default=128,\
                    help='dimention of GCN output, default: 128')
parser.add_argument('--numHeads', type=int, default=5,\
                    help='heads of multi-heads self-attention, default: 5')
parser.add_argument('--dropout', type=float, default=0.0,\
                    help='dropout rate for model, default: 0.0')
# dataset parameters
parser.add_argument('--dataPath', type=str, default='../dataset/semeval2017-task8/',\
                    help='path to training dataset, default: ../dataset/semeval2017-task8/')
parser.add_argument('--w2vPath', type=str, default='../dataset/glove/',\
                    help='path to word2vec dataset, default: ../dataset/glove/')
# train parameters
parser.add_argument('--optimizer', type=str, default='Adam',\
                    help='set optimizer type in [SGD/Adam], default: Adam')
parser.add_argument('--lr', type=float, default=1e-3,\
                    help='set learning rate, default: 0.001')
parser.add_argument('--weightDecay', type=float, default=1e-3,\
                    help='set weight decay for L2 Regularization, default: 0.001')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda/cpu), default: cuda')
parser.add_argument('--logName', type=str, default='./log/log.txt',\
                    help='log file name, default: log.txt')
parser.add_argument('--savePath', type=str, default='./model/model.pt',\
                    help='path to save model')
parser.add_argument('--task', type=int, default=0,\
                    help='select task to train, 0: both, 1: rumor, 2: stance, default: 0')

def main():
    args = parser.parse_args()
    
    # 获取数据集及词嵌入
    print('preparing data...', end='')
    sys.stdout.flush()
    dataset = semEval2017Dataset(
        dataPath = args.dataPath, 
        type = 'train'
    )
    loader = DataLoader(dataset, shuffle = True, num_workers = 4, collate_fn=collate)
    devDataset = semEval2017Dataset(
        dataPath = args.dataPath, 
        type = 'dev'
    )
    devLoader = DataLoader(devDataset, shuffle = True, num_workers = 4, collate_fn=collate)
    with open(args.dataPath + 'trainSet.json', 'r') as f:
            content = f.read()
    rawDataset = json.loads(content)
    label2IndexRumor = copy(rawDataset['label2IndexRumor'])
    label2IndexStance = copy(rawDataset['label2IndexStance'])
    del rawDataset

    word2vec = KeyedVectors.load_word2vec_format(
        '../dataset/glove/glove.twitter.27B.' + str(args.w2vDim) + 'd.gensim.txt',
        binary=False
    )
    vectorSize = word2vec.vector_size
    word2vec.add_vectors(["<start>", "<end>", "<unk>"] ,np.random.randn(3, vectorSize))
    with open('../dataset/semeval2017-task8/wordList.json', 'r') as f:
        content = f.read()
    wordList = ["<unk>", "<start>", "<end>"]
    wordList += (json.loads(content)).keys()
    word2index = {}
    index = 1
    for word in wordList:
        if word in word2vec:
            word2index[word] = index
            index += 1
    print('done.')

    # 选定实验设备
    f = open(args.logName, 'w')
    if 'cuda' in args.device:
        device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))
        if torch.cuda.is_available():
            trainInfo = 'train ABGCN on device:' + args.device + '\n'
        else:
            trainInfo = 'train ABGCN on device: cpu\n'
    else:
        device = torch.device('cpu')
        trainInfo = 'train ABGCN on device: cpu\n'
    f.write(trainInfo)
    print(trainInfo)
    f.close()
    
    # 声明模型、损失函数、优化器
    model = ABGCN(
        word2vec = word2vec,
        word2index = word2index,
        s2vDim = args.s2vDim,
        gcnHiddenDim = args.gcnHiddenDim,
        rumorFeatureDim = args.rumorFeatureDim,
        numRumorTag = len(label2IndexRumor),
        numStanceTag = len(label2IndexStance),
        numHeads = args.numHeads,
        dropout = args.dropout
    )
    model = model.set_device(device)
    print(model)
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    if args.optimizer == 'Adam':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = args.lr, 
            weight_decay = args.weightDecay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), 
            lr = args.lr, 
            momentum = 0.9,
            weight_decay = args.weightDecay
        )
    
    start = 1
    minLossRumor = float('inf')
    minLossStance = float('inf')
    maxMicroF1Rumor = float('-inf')
    maxMacroF1Rumor = float('-inf')
    maxMicroF1Stance = float('-inf')
    maxMacroF1Stance = float('-inf')
    maxAccuracyRumor = float('-inf')
    maxAccuracyStance = float('-inf')
    earlyStopCounter = 0

    for epoch in range(start, args.epoch + 1):
        f = open(args.logName, 'a')
        f.write('[epoch {:d}] '.format(epoch))

        # 训练模型
        if args.task == 1:
            task = 1
        elif args.task == 2: 
            task = 2
        else:
            task = randint(1, 2)
        f.write('working on task {:d}({:s})\n'.format(task, 'rumor detection' if task == 1 else 'stance classification'))
        true = []
        pre = []
        totalLoss = 0.
        
        model.train()
        for thread in tqdm(
            iter(loader), 
            desc="[epoch {:d}, {:s}]".format(epoch, 'rumor' if task == 1 else 'stance'), 
            leave=False, 
            ncols=100
        ):
            if task == 1:
                tag = thread['rumorTag'].to(device)
                true += thread['rumorTag'].tolist()
            else:
                tag = thread['stanceTag'].to(device)
                true += thread['stanceTag'].tolist()
            
            nodeText = thread['nodeText']
            for i in range(len(nodeText)):
                indexList = []
                for word in nodeText[i]:
                    if word in word2index:
                        indexList.append(word2index[word])
                    elif word != '':
                        indexList.append(word2index['<unk>'])
                nodeText[i] = torch.IntTensor(indexList).to(device)
            nodeText = pad_sequence(nodeText, padding_value=0, batch_first=True)
            thread['nodeText'] = nodeText

            optimizer.zero_grad()
            predict = model.forward(thread, task)
            loss = loss_func(predict, tag)
            totalLoss += loss
            loss.backward()
            optimizer.step()
            
            predict = softmax(predict, dim=1)
            pre += predict.max(dim=1)[1].tolist()
        macroF1 = f1_score(true, pre, labels=[0,1,2,3], average='macro')
        acc = (np.array(true) == np.array(pre)).sum() / len(true)
        f.write("average loss: {:.4f}, accuracy: {:.4f}, macro-f1: {:.4f}\n".format(
            totalLoss / len(loader), 
            acc,
            macroF1
        ))

        # 测试并保存模型
        if epoch % 5 == 0: # 每5个eopch进行一次测试，使用测试集数据
            f.write('==================================================\ntest model on dev set\n')
            # rumorTrue = []
            # rumorPre = []
            # totalLossRumor = 0.
            stanceTrue = []
            stancePre = []
            totalLossStance = 0.
            
            f.write('testing on both task\n')
            model.eval()
            for thread in tqdm(
                iter(devLoader), 
                desc="[epoch {:d}, test]".format(epoch), 
                leave=False, 
                ncols=80
            ):
                # rumorTag = data['rumorTag'].to(device)
                # rumorTrue += data['rumorTag'].tolist()
                stanceTag = thread['stanceTag'].to(device)
                stanceTrue += thread['stanceTag'].tolist()

                nodeText = thread['nodeText']
                for i in range(len(nodeText)):
                    indexList = []
                    for word in nodeText[i]:
                        if word in word2index:
                            indexList.append(word2index[word])
                        elif word != '':
                            indexList.append(word2index['<unk>'])
                    nodeText[i] = torch.IntTensor(indexList).to(device)
                nodeText = pad_sequence(nodeText, padding_value=0, batch_first=True)
                thread['nodeText'] = nodeText
                
                # pRumor = model.forwardRumor(data)
                # loss = loss_func(pRumor, rumorTag)
                # totalLossRumor += loss
                # pRumor = softmax(pRumor, 1)
                # rumorPre += pRumor.max(dim=1)[1].tolist()

                optimizer.zero_grad()
                predict = model.forward(thread, 2)
                loss = loss_func(predict, stanceTag)
                totalLossStance += loss
                predict = softmax(predict, dim=1)
                stancePre += predict.max(dim=1)[1].tolist()
            
            # microF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='micro')
            # macroF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='macro')
            # accuracyRumor = (np.array(rumorTrue) == np.array(rumorPre)).sum() / len(rumorPre)
            microF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='micro')
            macroF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='macro')
            accuracyStance = (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre)
            
#==============================================
# 保存rumor detection任务marco-F1最大时的模型
            # if(macroF1Rumor > maxMacroF1Rumor):
            #     model.save(args.savePath)
#==============================================
            # if totalLossRumor / len(devLoader) < minLossRumor:
            #     earlyStopCounter = 0
            # else:
            #     earlyStopCounter += 1

            # f.write('rumor detection:\n')
            # f.write('average loss: {:f}, accuracy: {:f}, micro-f1: {:f}, macro-f1: {:f}\n'.format(
            #     totalLossRumor / len(loader), accuracyRumor, microF1Rumor, macroF1Rumor
            # ))
            f.write('stance analyze:\n')
            f.write('average loss: {:f}, accuracy: {:f}, micro-f1: {:f}, macro-f1: {:f}\n'.format(
                totalLossStance / len(devLoader), 
                accuracyStance, 
                microF1Stance, 
                macroF1Stance
            ))

            # minLossRumor = min(minLossRumor, totalLossRumor / len(loader))
            # minLossStance = min(minLossStance, totalLossStance / len(testIndex))
            # maxMacroF1Rumor = max(maxMacroF1Rumor, macroF1Rumor)
            # maxMacroF1Stance = max(maxMacroF1Stance, macroF1Stance)
            # maxMicroF1Rumor = max(maxMicroF1Rumor, microF1Rumor)
            # maxMicroF1Stance = max(maxMicroF1Stance, microF1Stance)
            # maxAccuracyRumor = max(maxAccuracyRumor, accuracyRumor)
            # maxAccuracyStance = max(maxAccuracyStance, accuracyStance)
            f.write('========================================\n')
        f.close()
        # if earlyStopCounter == 6:
        #     print('early stop when loss on both task did not decrease')
        #     break
# end main()

if __name__ == '__main__':
    main()