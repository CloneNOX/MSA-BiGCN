import json
import torch
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch import optim
from torch.nn.functional import softmax
from model import *
from utils import *
from random import randint, shuffle
from tqdm import tqdm

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
parser.add_argument('--modelType', type=str, default='mtus',\
                    help='select model type between mtus and mtes, default: mtus')
parser.add_argument('--s2mType', type=str, default='cat',\
                    help='select way(add/cat) to get intput for GRU in dfferent mission, only used in mtes, default: cat')
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
parser.add_argument('--logPath', type=str, default='./log/log.txt',\
                    help='log file name, default: log.txt')
parser.add_argument('--savePath', type=str, default='./model/model.pt',\
                    help='path to save model')

def main():
    args = parser.parse_args()
    trainSet, label2IndexRumor, label2IndexStance, tfidf_vec = getTrainSet(args.dataPath)
    testSet, _, _ = getTestSet(args.dataPath, tfidf_vec)

    if args.modelType == 'mtus':
        model = MTUS(embeddingDim=args.embeddingDim, 
                     hiddenDim=args.hiddenDim,
                     inputDim=trainSet[0][0].size()[1], 
                     numGRULayer=args.gruLayer,
                     numRumorClass=len(label2IndexRumor),
                     numStanceClass=len(label2IndexStance))
    elif args.modelType == 'mtes':
        model = MTES(embeddingDim=args.embeddingDim, 
                     hiddenDim=args.hiddenDim,
                     inputDim=trainSet[0][0].size()[1], 
                     numGRULayer=args.gruLayer,
                     typeUS2M=args.s2mType,
                     numRumorClass=len(label2IndexRumor),
                     numStanceClass=len(label2IndexStance))
    else:
        return
    f = open(args.logPath, 'w')
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weightDecay, momentum=0.9) # 优化器，原文使用AdaGrad
    
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
        f = open(args.logPath, 'a')
        f.write('[epoch {:d}] '.format(epoch))
        shuffle(trainSet)# 打乱训练集的顺序            

        # 训练模型
        if randint(0, 1) % 2 == 0: # 训练M1
            f.write('working on task 1(rumor detection)\n')
            model.train()
            totalLoss = 0.
            rumorTrue = []
            rumorPre = []

            for i in tqdm(range(len(trainSet)), desc="[epoch {:d}, rumor]".format(epoch), leave=False, ncols=80):
                x = trainSet[i][0].to(device)
                rumorTag = trainSet[i][1].to(device)
                rumorTrue += trainSet[i][1].tolist()
                optimizer.zero_grad()
                
                p = model.forwardRumor(x)
                loss = loss_func(p, rumorTag)
                totalLoss += loss
                loss.backward()
                optimizer.step()
                
                p = softmax(p, 1)
                rumorPre += p.max(dim=1)[1].tolist()

            macroF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='macro')
            f.write("average loss: {:.4f}, accuracy: {:.4f}, marco F1: {:.4f}\n".format(
                totalLoss / len(trainSet), 
                (np.array(rumorTrue) == np.array(rumorPre)).sum() / len(rumorPre), 
                macroF1Rumor))
        else: # 训练M2
            f.write('working on task 2(stance analyze)\n')
            model.train()
            totalLoss = 0.
            stanceTrue = []
            stancePre = []

            for i in tqdm(range(len(trainSet)), desc="[epoch {:d}, stance]".format(epoch), leave=False, ncols=80):
                x = trainSet[i][0].to(device)
                stanceTag = trainSet[i][2].to(device)
                stanceTrue += trainSet[i][2].tolist()
                optimizer.zero_grad()
                
                p = model.forwardStance(x).view(-1, len(label2IndexStance))
                loss = loss_func(p, stanceTag)
                totalLoss += loss
                loss.backward()
                optimizer.step()
                
                p = softmax(p, 1)
                stancePre += p.max(dim=1)[1].tolist()
            
            macroF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='macro')
            f.write("average loss: {:.4f}, accuracy: {:.4f}, marco F1: {:.4f}\n".format(
                totalLoss / len(trainSet), 
                (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre),
                macroF1Stance))
        
        # 测试
        if epoch % 5 == 0: # 每5个eopch进行一次测试，使用测试集数据
            f.write('==========\nmodel testing\n')
            model.eval()
            rumorTrue = []
            stanceTrue = []
            rumorPre = []
            stancePre = []
            totalLossRumor = 0.
            totalLossStance = 0.
            
            f.write('testing on both task\n')
            for i in tqdm(range(len(testSet)), desc="[epoch {:d}, test]".format(epoch), leave=False, ncols=80):
                x = testSet[i][0].to(device)
                rumorTag = testSet[i][1].to(device)
                stanceTag = testSet[i][2].to(device)
                rumorTrue += testSet[i][1].tolist()
                stanceTrue += testSet[i][2].tolist()
                
                pRumor = model.forwardRumor(x)
                pStance = model.forwardStance(x).view(-1, len(label2IndexStance))
                loss = loss_func(pRumor, rumorTag)
                totalLossRumor += loss
                loss = loss_func(pStance, stanceTag)
                totalLossStance += loss
                pRumor = softmax(pRumor, 1)
                rumorPre += pRumor.max(dim=1)[1].tolist()
                pStance = softmax(pStance, 1)
                stancePre += pStance.max(dim=1)[1].tolist()
            
            microF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='micro')
            macroF1Rumor = f1_score(rumorTrue, rumorPre, labels=[0,1,2], average='macro')
            microF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='micro')
            macroF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='macro')
            accuracyRumor = (np.array(rumorTrue) == np.array(rumorPre)).sum() / len(rumorPre)
            accuracyStance = (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre)

#==============================================
# 保存rumor detection任务marco F1最大时的模型
            if(macroF1Rumor > maxMacroF1Rumor):
                model.save(args.savePath)
#==============================================
            if totalLossRumor / len(testSet) + macroF1Stance / len(testSet) < minLossRumor + minLossStance:
                earlyStopCounter = 0
            else:
                earlyStopCounter += 1

            f.write('rumor detection:\n')
            f.write('average loss: {:f}, accuracy: {:f}, micro-f1: {:f}, macro-f1: {:f}\n'.format(
                totalLossRumor / len(range(len(trainSet))), accuracyRumor, microF1Rumor, macroF1Rumor
            ))
            f.write('stance analyze:\n')
            f.write('average loss: {:f}, accuracy: {:f}, micro-f1: {:f}, macro-f1: {:f}\n'.format(
                totalLossStance / len(trainSet), accuracyStance, microF1Stance, macroF1Stance
            ))

            minLossRumor = min(minLossRumor, totalLossRumor / len(testSet))
            minLossStance = min(minLossStance, totalLossStance / len(testSet))
            maxMacroF1Rumor = max(maxMacroF1Rumor, macroF1Rumor)
            maxMacroF1Stance = max(maxMacroF1Stance, macroF1Stance)
            maxMicroF1Rumor = max(maxMicroF1Rumor, microF1Rumor)
            maxMicroF1Stance = max(maxMicroF1Stance, microF1Stance)
            maxAccuracyRumor = max(maxAccuracyRumor, accuracyRumor)
            maxAccuracyStance = max(maxAccuracyStance, accuracyStance)
            f.write('==========\n')
        f.close()
        if earlyStopCounter == 10:
            print('early stop when loss on both task did not decrease')
            break
# end main()
                
def getTrainSet(dataSetPath:str):
    with open(dataSetPath + 'trainSet.json', 'r') as f:
        content = f.read()
    trainSet = json.loads(content)
    threads = []
    rumorTags = []
    stanceTags = []
    for threadId in trainSet['threadIds']:
        thread = []
        stanceTag = []
        structure = trainSet['structures'][threadId]
        ids = flattenStructure(structure)
        time2Id = {}
        for id in ids:
            if id in trainSet['posts']:
                time2Id[str(trainSet['posts'][id]['time'])] = id
        # post按照时间先后排序
        time2Id = sorted(time2Id.items(), key=lambda d: d[0])
        for (time, id) in time2Id:
            if id in trainSet['posts'] and id in trainSet['stanceTag']:
                thread.append(trainSet['posts'][id]['text'])
                stanceTag.append(trainSet['label2IndexStance'][trainSet['stanceTag'][id]])
        threads.append(thread)
        rumorTags.append(torch.LongTensor([trainSet['label2IndexRumor'][trainSet['rumorTag'][threadId]]]))
        stanceTags.append(torch.LongTensor(stanceTag))
    cropus = []
    for thread in threads:
        for text in thread:
            cropus.append(text)
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(cropus).toarray()
    counter = 0
    for i in range(len(threads)):
        tfidf = []
        for _ in threads[i]:
            tfidf.append(tfidf_matrix[counter])
            counter += 1
        threads[i] = torch.Tensor(tfidf)

    label2IndexRumor = trainSet['label2IndexRumor']
    label2IndexStance = trainSet['label2IndexStance']
    trainSet = []
    for i in range(len(threads)):
        trainSet.append((threads[i], rumorTags[i], stanceTags[i]))
    return trainSet, label2IndexRumor, label2IndexStance, tfidf_vec
# end getTrainSet()

def getTestSet(dataPath: str, tfidf_vec):
    with open('../dataset/semeval2017-task8/' + 'testSet.json', 'r') as f:
        content = f.read()
    testSet = json.loads(content)
    threads = []
    rumorTags = []
    stanceTags = []
    for threadId in testSet['threadIds']:
        thread = []
        stanceTag = []
        structure = testSet['structures'][threadId]
        ids = flattenStructure(structure)
        time2Id = {}
        for id in ids:
            if id in testSet['posts']:
                time2Id[str(testSet['posts'][id]['time'])] = id
        # post按照时间先后排序
        time2Id = sorted(time2Id.items(), key=lambda d: d[0])
        for (time, id) in time2Id:
            if id in testSet['posts'] and id in testSet['stanceTag']:
                thread.append(testSet['posts'][id]['text'])
                stanceTag.append(testSet['label2IndexStance'][testSet['stanceTag'][id]])
        threads.append(thread)
        rumorTags.append(torch.LongTensor([testSet['label2IndexRumor'][testSet['rumorTag'][threadId]]]))
        stanceTags.append(torch.LongTensor(stanceTag))
    cropus = []
    for thread in threads:
        for text in thread:
            cropus.append(text)
    tfidf_matrix = tfidf_vec.transform(cropus).toarray()
    counter = 0
    for i in range(len(threads)):
        tfidf = []
        for _ in threads[i]:
            tfidf.append(tfidf_matrix[counter])
            counter += 1
        threads[i] = torch.Tensor(tfidf)

    label2IndexRumor = testSet['label2IndexRumor']
    label2IndexStance = testSet['label2IndexStance']
    testSet = []
    for i in range(len(threads)):
        testSet.append((threads[i], rumorTags[i], stanceTags[i]))
    return testSet, label2IndexRumor, label2IndexStance
# end getTestSet()


if __name__ == '__main__':
    main()