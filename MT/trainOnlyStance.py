import json
import torch
import argparse
import numpy as np
from data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from model import *
from utils import *
from random import randint, shuffle
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Training script for MTUS/MTES")

# model hyperparameters
parser.add_argument('--tfidfDim', type=int, default=5000,\
                    help='dimention of tfidf vectorizer, default: 5000')
parser.add_argument('--embeddingDim', type=int, default=100,\
                    help='dimention of embedding layer, default: 100')
parser.add_argument('--hiddenDim' ,type=int, default=100,\
                    help='dimention of hdiienDim, includs GRU layer and Linear layer, default: 100')
parser.add_argument('--gruLayer', type=int, default=1,\
                    help='num of GRU module layer, default: 1')
parser.add_argument('--bidirectional', type=bool, default=False,\
                    help='whether use bi-directional GRU, default: False')
parser.add_argument('--modelType', type=str, default='mtus',\
                    help='select model type between mtus and mtes, default: mtus')
parser.add_argument('--s2mType', type=str, default='add',\
                    help='select way(add/cat) to get intput for GRU in dfferent mission, only used in mtes, default: add')
# dataset parameters
parser.add_argument('--dataPath', type=str, default='../dataset/semeval2017-task8/',\
                    help='path to semeval2017 task8 dataset, default: ../dataset/semeval2017-task8/')
# train parameters
parser.add_argument('--lr', type=float, default=3e-4,\
                    help='set learning rate, default: 0.0003')
parser.add_argument('--weightDecay', type=float, default=5e-4,\
                    help='set weight decay for L2 Regularization, default: 0.0005')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda:0/cpu), default: cuda')
parser.add_argument('--logName', type=str, default='./log/log.txt',\
                    help='log file name, default: log.txt')
parser.add_argument('--savePath', type=str, default='./model/model.pt',\
                    help='path to save model')

def main():
    args = parser.parse_args()
    
    tfidf_vec, label2IndexRumor, label2IndexStance = getTfidfAndLabel(args.dataPath, args.tfidfDim)
    inputDim = tfidf_vec.max_features

    if 'PHEME' in args.dataPath:
        datasetType = 'PHEME'
    else:
        datasetType = 'semEval'
    if datasetType == 'semEval':
        trainSet = semEval2017Dataset(
            dataPath = '../dataset/semeval2017-task8/', 
            type = 'train'
        )
        testSet = semEval2017Dataset(
            dataPath = '../dataset/semeval2017-task8/',
            type = 'test'
        )
    elif datasetType == 'PHEME':
        trainSet = PHEMEDataset(
            dataPath = args.dataPath, 
            type = 'train'
        )
        testSet = PHEMEDataset(
            dataPath = args.dataPath, 
            type = 'test'
        )
    trainLoader = DataLoader(trainSet, collate_fn=collate, shuffle=True)
    testLoader = DataLoader(testSet, collate_fn=collate, shuffle=True)
    
    if args.modelType == 'mtus':
        model = MTUS(
            embeddingDim=args.embeddingDim, 
            hiddenDim=args.hiddenDim,
            inputDim=inputDim, 
            numGRULayer=args.gruLayer,
            numRumorClass=len(label2IndexRumor),
            numStanceClass=len(label2IndexStance)
        )
    elif args.modelType == 'mtes':
        model = MTES(
            embeddingDim=args.embeddingDim, 
            hiddenDim=args.hiddenDim,
            inputDim=inputDim, 
            numGRULayer=args.gruLayer,
            numRumorClass=len(label2IndexRumor),
            numStanceClass=len(label2IndexStance)
        )
    else:
        return
    f = open(args.logName, 'w')
    if 'cuda' in args.device:
        device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))
        if torch.cuda.is_available():
            f.write('train with model: ' + args.modelType + ' on device:' + args.device + '\n')
        else:
            f.write('train with model: ' + args.modelType + ' on device: cpu\n')
    else:
        device = torch.device('cpu')
        f.write('train with model: ' + args.modelType + ' on device: cpu\n')
    f.close()
    model = model.set_device(device)
    print(model)

    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, 
                            weight_decay=args.weightDecay) # 优化器，原文使用AdaGrad
    
    start = 1
    maxMacroF1Stance = float('-inf')
    minTestLoss = float('inf')
    earlyStopCounter = 0

    testStanceAcc = []
    testStanceF1 = []
    testLoss = []

    for epoch in range(start, args.epoch + 1):
        f = open(args.logName, 'a')
        f.write('[epoch {:d}] '.format(epoch))          

        # 训练模型
        f.write('working on task 2(stance analyze)\n')
        model.train()
        totalLoss = 0.
        stanceTrue = []
        stancePre = []

        for thread in tqdm(iter(trainLoader),
                            desc="[epoch {:d}, stance]".format(epoch), 
                            leave=False, 
                            ncols=80):
            posts = torch.Tensor(tfidf_vec.transform(thread['thread']).toarray()).to(device)
            stanceTrue += thread['stanceTag'].tolist()
            stanceTag = thread['stanceTag'].to(device)
            optimizer.zero_grad()
            
            p = model.forwardStance(posts)
            loss = loss_func(p, stanceTag)
            totalLoss += loss
            loss.backward()
            optimizer.step()
            
            p = softmax(p, 1)
            stancePre += p.max(dim=1)[1].tolist()
        
        macroF1Stance = f1_score(stanceTrue, stancePre, labels=range(len(label2IndexStance)), average='macro')
        f.write("   stance classification average loss: {:.4f}, accuracy: {:.4f}, marco F1: {:.4f}\n".format(
            totalLoss / len(trainSet), 
            (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre),
            macroF1Stance))
        
        # 测试
        if epoch % 5 == 0: # 每5个eopch进行一次测试，使用测试集数据
            f.write('==========\nmodel testing\n')
            model.eval()
            stanceTrue = []
            stancePre = []
            totalLossStance = 0.
            
            f.write('testing on both task\n')
            for thread in tqdm(iter(testLoader), 
                               desc="[epoch {:d}, test]".format(epoch), 
                               leave=False, 
                               ncols=80):
                posts = torch.Tensor(tfidf_vec.transform(thread['thread']).toarray()).to(device)
                stanceTrue += thread['stanceTag'].tolist()
                stanceTag = thread['stanceTag'].to(device)
                
                pRumor = model.forwardRumor(posts)
                pStance = model.forwardStance(posts)
                loss = loss_func(pStance, stanceTag)
                totalLossStance += loss
                pStance = softmax(pStance, 1)
                stancePre += pStance.max(dim=1)[1].tolist()
            
            macroF1Stance = f1_score(stanceTrue, stancePre, labels=range(len(label2IndexStance)), average='macro')
            accuracyStance = (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre)

#==============================================
# 保存rumor detection任务marco F1最大时的模型
            if(macroF1Stance > maxMacroF1Stance):
                model.save(args.savePath)
                saveStatus = {
                    'epoch': epoch,
                    'macroF1Stance': macroF1Stance,
                    'accStance': accuracyStance,
                    'loss': (totalLossStance / len(testLoader)).item()
                }
                maxMacroF1Stance = max(maxMacroF1Stance, macroF1Stance)
                f.write('saved model\n')
            else:
                earlyStopCounter += 1
#==============================================
            f.write('stance analyze:\n')
            f.write('average loss: {:f}, accuracy: {:f}, macro-f1: {:f}\n'.format(
                totalLossStance / len(trainSet), accuracyStance, macroF1Stance
            ))
            f.write('early stop counter: {:d}\n'.format(earlyStopCounter))
            f.write('==========\n')
            testLoss.append((totalLossStance / len(testLoader)).item())
            testStanceAcc.append(accuracyStance)
            testStanceF1.append(macroF1Stance)
        f.close()
        if earlyStopCounter == 10:
            print('early stop when loss on both task did not decrease')
            break
    print(saveStatus)
    with open(args.logName, 'a') as f:
        f.write('\n\nsave model at epoch {:d},\
                \nmarco F1 stance: {:f}, acc stance: {:f}\n'.format(
                    saveStatus['epoch'],
                    saveStatus['macroF1Stance'], saveStatus['accStance']
                ))
    
    saveStatus['testStanceAcc'] = testStanceAcc
    saveStatus['testStanceF1'] = testStanceF1
    saveStatus['testLoss'] = testLoss
    with open(args.logName[:-4] + '.json', 'w') as f:
        f.write(json.dumps(saveStatus))
# end main()

if __name__ == '__main__':
    main()