import json
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch import optim
from torch.nn.functional import softmax
import argparse
from model import *
from defaultParameter import*
from utils import *
from random import randint, shuffle

parser = argparse.ArgumentParser(description="Training script for MTUS/MTES")

# model hyperparameters
parser.add_argument('--embeddingDim', type=int, default=100,\
                    help='dimention of embedding layer, default: 100')
parser.add_argument('--hiddenDim' ,type=int, default=100,\
                    help='dimention of hdiienDim, includs GRU layer and Linear layer, default: 100')
parser.add_argument('--gruLayer', type=int, default=2,\
                    help='num of GRU module layer, default: 2')
parser.add_argument('--bidirectional', type=bool, default=False,\
                    help='whether use bi-directional GRU, default: False')
parser.add_argument('--modelType', type=str, default='mtes',\
                    help='select model type between mtus and mtes, default: mtes')
# dataset parameters
parser.add_argument('--dataPath', type=str, default='../dataset/semeval2017-task8/',\
                    help='path to semeval2017 task8 dataset, default: ../dataset/semeval2017-task8/')
# train parameters
parser.add_argument('--lr', type=float, default=1e-3,\
                    help='set learning rate, default: 0.001')
parser.add_argument('--weightDecay', type=float, default=1e-2,\
                    help='set weight decay for L2 Regularization, default: 0.01')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cpu',\
                    help='select device(cuda:0/cpu), default: cpu')


def main():
    args = parser.parse_args()
    trainSet, label2IndexRumor, label2IndexStance = getTrainSet(args.dataPath)

    if args.modelType == 'mtus':
        model = MTUS(embeddingDim=args.embeddingDim, hiddenDim=args.hiddenDim,\
                     inputDim=trainSet[0][0].size()[1], numGRULayer=args.gruLayer,\
                     numRumorClass=len(label2IndexRumor),\
                     numStanceClass=len(label2IndexStance))
    elif args.modelType == 'mtes':
        model = MTES(embeddingDim=args.embeddingDim, hiddenDim=args.hiddenDim,\
                     inputDim=trainSet[0][0].size()[1], numGRULayer=args.gruLayer,\
                     numRumorClass=len(label2IndexRumor),\
                     numStanceClass=len(label2IndexStance))
    else:
        return
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weightDecay) # 优化器，原文使用AdaGrad
    if 'cuda' in args.device:
        device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))
        print('train with model: ' + args.modelType + 'on device: cuda')
    else:
        device = torch.device('cpu')
        print('train with model: ' + args.modelType + 'on device: cpu')
    model = model.set_device(device)
    print(model)
    
    start = 1
    minLossRumor = float('inf')
    maxMicroF1Rumor = float('-inf')
    maxMacroF1Rumor = float('-inf')
    minLossStance = float('inf')
    maxMicroF1Stance = float('-inf')
    maxMacroF1Stance = float('-inf')
    for epoch in range(start, args.epoch + 1):
        print('[epoch {:>3d}]'.format(epoch), end=" ")
        shuffle(trainSet)# 打乱训练集的顺序            
        totalLoss = 0.

        # 训练模型
        if randint % 2 == 0: # 训练M1
            print('working on task 1(rumor detection)')
            model.train()
            for i in range(len(trainSet)):
                x = trainSet[i][0].to(device).to(device)
                rumorTag = trainSet[i][1].to(device)
                optimizer.zero_grad()
                
                p = model.forwardRumor(x)
                loss = loss_func(p, rumorTag)
                totalLoss += loss
                loss.backward()
                optimizer.step()
            print("average loss: {:.4f}".format(totalLoss / len(trainSet)))
        else: # 训练M2
            print('working on task 2(stance analyze)')
            model.train()
            totalLoss = 0.

            for i in range(len(trainSet)):
                x = trainSet[i][0].to(device)
                stanceTag = trainSet[i][2].to(device)
                optimizer.zero_grad()
                
                p = model.forwardStance(x).view(-1, len(label2IndexStance))
                loss = loss_func(p, stanceTag)
                totalLoss += loss
                loss.backward()
                optimizer.step()
            print("average loss: {:.4f}".format(totalLoss / len(trainSet)))
        
        # 测试
        if epoch % 5 == 0: # 每5个eopch进行一次测试，随机抽取训练集中的数据
            print('==========\nmodel testing')
            model.eval()
            testIndex = [randint(0, len(trainSet)) for i in range(len(trainSet) // 5)]
            rumorTrue = []
            stanceTrue = []
            rumorPre = []
            stancePre = []
            totalLossRumor = 0.
            totalLossStance = 0.
            
            print('testing on both task')
            for i in testIndex:
                x = trainSet[i][0].to(device)
                rumorTag = trainSet[i][1].to(device)
                stanceTag = trainSet[i][2].to(device)
                rumorTrue += trainSet[i][1].tolist()
                stanceTrue = trainSet[i][2].tolist()
                
                pRumor = model.forwardRumor(x)
                pStance = model.forwardStance(x).view(-1, len(trainSet['label2IndexStance']))
                loss = loss_func(pRumor, rumorTag)
                totalLossRumor += loss
                loss = loss_func(pStance, stanceTag)
                totalLossStance += loss
                pRumor = softmax(pRumor, 1)
                rumorPre += pRumor.max(dim=1)[1].tolise()
                pStance = softmax(pStance, 1)
                stancePre += pStance.max(dim=1)[1].tolist()
            
            microF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='micro')
            macroF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='macro')
            microF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2], average='micro')
            macroF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2], average='macro')
            print('rumor detection:')
            print('average loss: {:f}, micro-f1: {:f}, macro-f1: {:f}'.format(
                totalLossRumor / len(testIndex), microF1Rumor, macroF1Rumor
            ))
            print('rumor analyze:')
            print('average loss: {:f}, micro-f1: {:f}, macro-f1: {:f}'.format(
                totalLossStance / len(testIndex), microF1Stance, macroF1Stance
            ))
            minLossRumor = min(minLossRumor, totalLossRumor / len(testIndex))
            minLossStance = min(minLossStance, totalLossStance / len(testIndex))
            maxMacroF1Rumor = max(maxMacroF1Rumor, macroF1Rumor)
            maxMacroF1Stance = max(maxMacroF1Stance, macroF1Stance)
            maxMicroF1Rumor = max(maxMicroF1Rumor, microF1Rumor)
            maxMicroF1Stance = max(maxMicroF1Stance, microF1Stance)
            print('==========')
    print('====================')
    print('rumor analyze:')
    print('min loss: {:f}, max micro-f1: {:f}, max macro-f1: {:f}'.format(
        minLossRumor, maxMicroF1Rumor, maxMacroF1Rumor
    ))
    print('rumor analyze:')
    print('average loss: {:f}, micro-f1: {:f}, macro-f1: {:f}'.format(
        minLossStance, maxMicroF1Stance, maxMacroF1Stance
    ))

                
            

def getTrainSet(dataSetPath:str):
    with open(dataSetPath + 'trainSet.json', 'r') as f:
        content = f.read()
    trainSet = json.loads(content)
    threads = []
    rumorTags = []
    stanceTags = []
    for threadId in trainSet['threadIds']:
        thread = {str(trainSet['posts'][threadId]['time']): trainSet['posts'][threadId]['text']}
        structure = trainSet['structures'][threadId]
        ids = flattenStructure(structure)
        stanceTag = []
        for id in ids:
            if id in trainSet['posts']:
                thread[str(trainSet['posts'][id]['time'])] = trainSet['posts'][id]['text']
                stanceTag.append(trainSet['label2IndexStance'][trainSet['stanceTag'][id]])
        # post按照时间先后排序
        thread = sorted(thread.items(), key=lambda d: d[0])
        threads.append(thread)
        rumorTags.append(torch.LongTensor([trainSet['label2IndexRumor'][trainSet['rumorTag'][threadId]]]))
        stanceTags.append(torch.LongTensor(stanceTag))
    cropus = []
    for thread in threads:
        for _, text in thread:
            cropus.append(text)
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(cropus).toarray()
    counter = 0
    for i in range(len(threads)):
        tfidf = []
        for _, _ in threads[i]:
            tfidf.append(tfidf_matrix[counter])
            counter += 1
        threads[i] = torch.Tensor(tfidf)

    label2IndexRumor = trainSet['label2IndexRumor']
    label2IndexStance = trainSet['label2IndexStance']
    trainSet = []
    for i in range(len(threads)):
        trainSet.append((threads[i], rumorTags[i], stanceTags[i]))
    return trainSet, label2IndexRumor, label2IndexStance


if __name__ == '__main__':
    main()