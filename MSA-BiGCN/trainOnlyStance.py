import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import argparse
import numpy as np
from data import *
from MSABiGCN import *
from sklearn.metrics import f1_score
from utils import *
from tqdm import tqdm
from copy import copy
import sys

parser = argparse.ArgumentParser(description="Training script for ABGCN")

# model hyperparameters
parser.add_argument('--w2vDim', type=int, default=50,\
                    help='dimention of word2vec, default: 50')
parser.add_argument('--s2vDim' ,type=int, default=64,\
                    help='dimention of sentence2vec(get from lstm/attention)')
parser.add_argument('--w2vAttHeads', type=int, default=1,\
                    help='heads of multi-heads self-attention in word2vec layer, default: 1')
parser.add_argument('--s2vAttHeads', type=int, default=1,\
                    help='heads of multi-heads self-attention in sentence2vec layer, default: 1')
parser.add_argument('--gcnHiddenDim', type=int, default=256,\
                    help='dimention of GCN hidden layer, default: 256')
parser.add_argument('--rumorFeatureDim', type=int, default=256,\
                    help='dimention of GCN output, default: 256')
parser.add_argument('--dropout', type=float, default=0.0,\
                    help='dropout rate for model, default: 0.0')
parser.add_argument('--needStance', type=str, default='True',\
                    help='whether use stance feature in rumor detection, default: True')
# dataset parameters
parser.add_argument('--dataPath', type=str, default='../dataset/semeval2017-task8/',\
                    help='path to training dataset, default: ../dataset/semeval2017-task8/')
parser.add_argument('--w2vPath', type=str, default='../dataset/glove/',\
                    help='path to word2vec dataset, default: ../dataset/glove/')
# train parameters
parser.add_argument('--optimizer', type=str, default='Adam',\
                    help='set optimizer type in [SGD/Adam], default: Adam')
parser.add_argument('--lr', type=float, default=3e-5,\
                    help='set learning rate, default: 3e-5')
parser.add_argument('--weightDecay', type=float, default=5e-4,\
                    help='set weight decay for L2 Regularization, default: 5e-4')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda/cpu), default: cuda')
parser.add_argument('--logName', type=str, default='./log/log.txt',\
                    help='log file name, default: ./log/log.txt')
parser.add_argument('--savePath', type=str, default='./model/model.pt',\
                    help='path to save model, default: ./model/model.txt')
parser.add_argument('--lossRatio', type=float, default=1.,\
                    help='ratio for loss-on-rumor:loss-on-stance, default: 1.0')

def main():
    args = parser.parse_args()
    
    # 获取数据集及词嵌入
    print('preparing data...', end='')
    sys.stdout.flush()
    dataset = semEval2017Dataset(
        dataPath = args.dataPath, 
        type = 'train'
    )
    testDataset = semEval2017Dataset(
        dataPath = args.dataPath, 
        type = 'test'
    )
    loader = DataLoader(dataset, shuffle = True, num_workers = 4, collate_fn=collate) 
    testLoader = DataLoader(testDataset, shuffle = True, num_workers = 4, collate_fn=collate)
    with open(args.dataPath + 'trainSet.json', 'r') as f:
            content = f.read()
    rawDataset = json.loads(content)
    label2IndexStance = copy(rawDataset['label2IndexStance'])
    del rawDataset

    word2vec = KeyedVectors.load_word2vec_format(
        '../dataset/glove/glove.twitter.27B.' + str(args.w2vDim) + 'd.gensim.txt',
        binary=False
    )
    vectorSize = word2vec.vector_size
    word2vec.add_vectors(["<start>", "<end>", "<unk>"] ,np.random.randn(3, vectorSize))
    with open(args.dataPath + 'wordList.json', 'r') as f:
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
            trainInfo = 'train MSABiGCN on device:' + args.device + '\n'
        else:
            trainInfo = 'train MSABiGCN on device: cpu\n'
    else:
        device = torch.device('cpu')
        trainInfo = 'train MSABiGCN on device: cpu\n'
    f.write(trainInfo)
    print(trainInfo)
    f.close()
    
    # 声明模型、损失函数、优化器
    model = MSAOnlyStance(
        word2vec = word2vec,
        word2index = word2index,
        s2vDim = args.s2vDim,
        numStanceTag = len(label2IndexStance),
        w2vAttentionHeads = args.w2vAttHeads,
        s2vAttentionHeads = args.s2vAttHeads,
        dropout = args.dropout
    )
    model = model.set_device(device)
    print(model)
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    # GCNParams = list(map(id, model.biGCN.parameters()))
    # baseParams = filter(lambda p:id(p) not in GCNParams, model.parameters())
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
    # 记录验证集上的最好性能，用于early stop
    earlyStopCounter = 0
    maxF1train = 0.
    maxF1test = 0.

    trainStanceAcc = []
    trainStanceF1 = []
    trainLoss = []

    testStanceAcc = []
    testStanceF1 = []
    testLoss = []

    for epoch in range(start, args.epoch + 1):
        f = open(args.logName, 'a')
        f.write('[epoch: {:d}] '.format(epoch))

        # 训练模型
        stanceTrue = []
        stancePre = []
        totalLoss = 0.
        
        model.train()
        for thread in tqdm(
            iter(loader), 
            desc="[epoch: {:d}] ".format(epoch), 
            leave=False, 
            ncols=100
        ):
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

            optimizer.zero_grad()
            stancePredict = model.forward(thread)
            loss = loss_func(stancePredict, stanceTag)
            totalLoss += loss
            loss.backward()
            optimizer.step()
            
            stancePredict = softmax(stancePredict, dim=1)
            stancePre += stancePredict.max(dim=1)[1].tolist()
        f.write('average loss: {:f}\n'.format(
            totalLoss / len(loader)
        ))
        macroF1 = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='macro')
        acc = (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stanceTrue)
        f.write("    stance classification accuracy: {:.4f}, macro-f1: {:.4f}\n".format(
            acc,
            macroF1
        ))
        trainStanceAcc.append(acc)
        trainStanceF1.append(macroF1)
        maxF1train = max(maxF1train, macroF1)
    
        # 测试并保存模型
        if epoch % 5 == 0: # 每1个eopch进行一次测试，使用测试集数据
            f.write('==================================================\ntest model on test set\n')
            stanceTrue = []
            stancePre = []
            totalLoss = 0.

            model.eval()
            for thread in tqdm(
                iter(testLoader), 
                desc="[epoch: {:d}, test]".format(epoch), 
                leave=False, 
                ncols=80
            ):
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
                
                stancePredict = model.forward(thread)
                loss = loss_func(stancePredict, stanceTag)
                totalLoss += loss

                stancePredict = softmax(stancePredict, dim=1)
                stancePre += stancePredict.max(dim=1)[1].tolist()
            
            macroF1Stance = f1_score(stanceTrue, stancePre, labels=[0,1,2,3], average='macro')
            accStance = (np.array(stanceTrue) == np.array(stancePre)).sum() / len(stancePre)
            
#==============================================
# 保存验证集marco F1和最大时的模型
            if macroF1Stance > maxF1test:
                model.save(args.savePath)
                earlyStopCounter = 0
                saveStatus = {
                    'epoch': epoch,
                    'macroF1Stance': macroF1Stance,
                    'accStance': accStance,
                    'loss': (totalLoss / len(testLoader)).item()
                }
                maxF1test = max(maxF1test, macroF1Stance)
                f.write('saved model\n')
            else:
                earlyStopCounter += 1
#==============================================
            f.write('average joint-loss: {:f}\n'.format(totalLoss / len(testLoader)))
            f.write('stance analyze:\n')
            f.write('accuracy: {:f}, macro-f1: {:f}\n'.format(
                accStance, 
                macroF1Stance
            ))
            f.write('early stop counter: {:d}\n'.format(earlyStopCounter))
            f.write('========================================\n')
            testLoss.append((totalLoss / len(testLoader)).item())
            testStanceAcc.append(accStance)
            testStanceF1.append(macroF1Stance)
        f.close()
        if earlyStopCounter >= 20: # 验证集上连续多次测试性能没有提升就早停
            print('early stop when F1 on dev set did not increase')
            break
    print(saveStatus)
    with open(args.logName, 'a') as f:
        f.write('\n\nsave model at epoch {:d},\
                \nmarco F1 stance: {:f}, acc stance: {:f}\n'.format(
                    saveStatus['epoch'],
                    saveStatus['macroF1Stance'], saveStatus['accStance']
                ))

    saveStatus['trainStanceAcc'] = trainStanceAcc
    saveStatus['trainStanceF1'] = trainStanceF1
    saveStatus['trainLoss'] = trainLoss

    saveStatus['testStanceAcc'] = testStanceAcc
    saveStatus['testStanceF1'] = testStanceF1
    saveStatus['testLoss'] = testLoss
    with open(args.logName[:-4] + '.json', 'w') as f:
        f.write(json.dumps(saveStatus))
# end main()

if __name__ == '__main__':
    main()