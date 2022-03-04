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
from sklearn.feature_extraction.text import TfidfVectorizer
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
parser.add_argument('--s2v', type=str, default='a',\
                    help='select method to get sentence2vec, a: attention, l: lstm, default: a')
parser.add_argument('--numLstmLayer', type=int, default=1,\
                    help='number of layers in lstm, only works when set: --s2v l, default: 1')
# dataset parameters
parser.add_argument('--dataPath', type=str, default='../dataset/semeval2017-task8/',\
                    help='path to training dataset, default: ../dataset/semeval2017-task8/')
parser.add_argument('--w2vPath', type=str, default='../dataset/glove/',\
                    help='path to word2vec dataset, default: ../dataset/glove/')
# train parameters
parser.add_argument('--lr', type=float, default=1e-3,\
                    help='set learning rate, default: 0.001')
parser.add_argument('--weightDecay', type=float, default=1e-2,\
                    help='set weight decay for L2 Regularization, default: 0.01')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cpu',\
                    help='select device(cuda/cpu), default: cpu')
parser.add_argument('--logName', type=str, default='./log/log.txt',\
                    help='log file name, default: log.txt')
parser.add_argument('--savePath', type=str, default='./model/model.pt',\
                    help='path to save model')

def main():
    args = parser.parse_args()
    
    # 获取数据集
    print('preparing data...', end='')
    sys.stdout.flush()
    dataset = semeval2017Dataset(dataPath = args.dataPath, 
                                 type = 'train',
                                 w2vPath = args.w2vPath,
                                 w2vDim = args.w2vDim)
    loader = DataLoader(dataset, shuffle=True)
    testDataset = semeval2017Dataset(dataPath = args.dataPath, 
                                     type = 'test',
                                     w2vPath = args.w2vPath,
                                     w2vDim = args.w2vDim)
    testLoader = DataLoader(testDataset, shuffle=True)
    with open(args.dataPath + 'trainSet.json', 'r') as f:
            content = f.read()
    rawDataset = json.loads(content)
    label2IndexRumor = copy(rawDataset['label2IndexRumor'])
    label2IndexStance = copy(rawDataset['label2IndexStance'])
    del rawDataset
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
    model = ABGCN(w2vDim = args.w2vDim,
                  s2vDim = args.s2vDim,
                  gcnHiddenDim = args.gcnHiddenDim,
                  rumorFeatureDim = args.rumorFeatureDim,
                  numRumorTag = len(label2IndexRumor),
                  numStanceTag = len(label2IndexStance),
                  s2vMethon = 'l')
    model = model.set_device(device)
    print(model)
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
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
        if randint(0, 0) % 2 == 0: # 训练M1
            f.write('working on task 1(rumor detection)\n')
            rumorTrue = []
            rumorPre = []
            totalLoss = 0.
            
            model.train()
            for data in tqdm(iter(loader), desc="[epoch {:d}, rumor]".format(epoch), leave=False, ncols=100):
                # 抹除dataloader生成batch时对数据的升维
                data['threadId'] = data['threadId'][0]
                data['threadIndex'] = data['threadIndex'][0]
                for i in range(len(data['nodeFeature'])):
                    data['nodeFeature'][i] = data['nodeFeature'][i].view((data['nodeFeature'][i].shape[1], data['nodeFeature'][i].shape[2]))
                data['edgeIndexTD'] = data['edgeIndexTD'].view((data['edgeIndexTD'].shape[1], data['edgeIndexTD'].shape[2]))
                data['edgeIndexBU'] = data['edgeIndexBU'].view((data['edgeIndexBU'].shape[1], data['edgeIndexBU'].shape[2]))
                data['rumorTag'] = data['rumorTag'].view((data['rumorTag'].shape[1]))
                data['stanceTag'] = data['stanceTag'].view((data['stanceTag'].shape[1]))
                rumorTag = data['rumorTag'].to(device)
                rumorTrue += data['rumorTag'].tolist()

                optimizer.zero_grad()
                p = model.forwardRumor(data)
                loss = loss_func(p, rumorTag)
                totalLoss += loss
                loss.backward()
                optimizer.step()
                
                p = softmax(p, dim=1)
                rumorPre += p.max(dim=1)[1].tolist()
            macroF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='macro')
            f.write("average loss: {:.4f}, macro-f1: {:.4f}\n".format(totalLoss / len(loader), macroF1Rumor))
        else: # 训练M2
            # f.write('working on task 2(stance analyze)\n')
            # model.train()
            # totalLoss = 0.

            # for i in tqdm(range(len(trainSet)), desc="[epoch {:d}, stance]".format(epoch), leave=False, ncols=80):
            #     x = trainSet[i][0].to(device)
            #     stanceTag = trainSet[i][2].to(device)
            #     optimizer.zero_grad()
                
            #     p = model.forwardStance(x).view(-1, len(label2IndexStance))
            #     loss = loss_func(p, stanceTag)
            #     totalLoss += loss
            #     loss.backward()
            #     optimizer.step()
            # f.write("average loss: {:.4f}\n".format(totalLoss / len(trainSet)))
            pass
        
        # 测试并保存模型
        if epoch % 5 == 0: # 每5个eopch进行一次测试，使用测试集数据
            f.write('==========\nsaved model testing\n')
            model.eval()
            rumorTrue = []
            rumorPre = []
            totalLossRumor = 0.
            stanceTrue = []
            stancePre = []
            totalLossStance = 0.
            
            f.write('testing on both task\n')
            for data in tqdm(iter(testLoader), desc="[epoch {:d}, test]".format(epoch), leave=False, ncols=100):
                # 抹除dataloader生成batch时对数据的升维
                data['threadId'] = data['threadId'][0]
                data['threadIndex'] = data['threadIndex'][0]
                for i in range(len(data['nodeFeature'])):
                    data['nodeFeature'][i] = data['nodeFeature'][i].view((data['nodeFeature'][i].shape[1], data['nodeFeature'][i].shape[2]))
                data['edgeIndexTD'] = data['edgeIndexTD'].view((data['edgeIndexTD'].shape[1], data['edgeIndexTD'].shape[2]))
                data['edgeIndexBU'] = data['edgeIndexBU'].view((data['edgeIndexBU'].shape[1], data['edgeIndexBU'].shape[2]))
                data['rumorTag'] = data['rumorTag'].view((data['rumorTag'].shape[1]))
                data['stanceTag'] = data['stanceTag'].view((data['stanceTag'].shape[1]))
                rumorTag = data['rumorTag'].to(device)
                stanceTag = data['stanceTag'].to(device)
                rumorTrue += data['rumorTag'].tolist()
                stanceTrue += data['stanceTag'].tolist()
                
                pRumor = model.forwardRumor(data)
                # stanceFeature = model.forwardStance(x).view(-1, len(label2IndexStance))
                loss = loss_func(pRumor, rumorTag)
                totalLossRumor += loss
                # loss = loss_func(pStance, stanceTag)
                # totalLossStance += loss
                pRumor = softmax(pRumor, 1)
                rumorPre += pRumor.max(dim=1)[1].tolist()
                # pStance = softmax(pStance, 1)
                # stancePre += pStance.max(dim=1)[1].tolist()
            
            microF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='micro')
            macroF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='macro')
            # microF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='micro')
            # macroF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='macro')
            accuracyRumor = (np.array(rumorTrue) == np.array(rumorPre)).sum() / len(rumorPre)
            # accuracyStance = (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre)
            
#==============================================
# 保存rumor detection任务marco-F1最大时的模型
            if(macroF1Rumor > maxMacroF1Rumor):
                model.save(args.savePath)
#==============================================
            if totalLossRumor / len(testLoader) < minLossRumor:
                earlyStopCounter = 0
            else:
                earlyStopCounter += 1

            f.write('rumor detection:\n')
            f.write('average loss: {:f}, accuracy: {:f}, micro-f1: {:f}, macro-f1: {:f}\n'.format(
                totalLossRumor / len(loader), accuracyRumor, microF1Rumor, macroF1Rumor
            ))
            # f.write('stance analyze:\n')
            # f.write('average loss: {:f}, accuracy: {:f}, micro-f1: {:f}, macro-f1: {:f}\n'.format(
            #     totalLossStance / len(testIndex), accuracyStance, microF1Stance, macroF1Stance
            # ))

            minLossRumor = min(minLossRumor, totalLossRumor / len(loader))
            # minLossStance = min(minLossStance, totalLossStance / len(testIndex))
            maxMacroF1Rumor = max(maxMacroF1Rumor, macroF1Rumor)
            # maxMacroF1Stance = max(maxMacroF1Stance, macroF1Stance)
            maxMicroF1Rumor = max(maxMicroF1Rumor, microF1Rumor)
            # maxMicroF1Stance = max(maxMicroF1Stance, microF1Stance)
            maxAccuracyRumor = max(maxAccuracyRumor, accuracyRumor)
            # maxAccuracyStance = max(maxAccuracyStance, accuracyStance)
            f.write('==========\n')
        f.close()
        if earlyStopCounter == 10:
            print('early stop when loss on both task did not decrease')
            break
# end main()

if __name__ == '__main__':
    main()