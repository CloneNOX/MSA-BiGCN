import torch
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
from MSABiGCN import MSABiGCNOnlyRumor
from transformers import BertTokenizer, BertModel
from rumorDataset import RumorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from copy import copy
from time import process_time
import sys
import logging

parser = argparse.ArgumentParser(description="Training script for ABGCN")

# model hyperparameters
parser.add_argument('--bert_path' ,type=str, default='../model/bert-base-cased',\
                    help="path to bert model, default: '../model/bert-base-cased'")
parser.add_argument('--embedding_dim' ,type=int, default=768,\
                    help='dimention of sentence2vec(get from lstm/attention)')
parser.add_argument('--GCN_hidden_dim', type=int, default=768,\
                    help='dimention of GCN hidden layer, default: 256')
parser.add_argument('--rumor_feature_dim', type=int, default=768,\
                    help='dimention of GCN output, default: 256')
parser.add_argument('--dropout', type=float, default=0.0,\
                    help='dropout rate for model, default: 0.0')
# dataset parameters
parser.add_argument('--data_path', type=str, default='../datasets/PHEME/',\
                    help='path to training dataset, default: ../datasets/PHEME/')
# train parameters
parser.add_argument('--optimizer', type=str, default='AdamW',\
                    help='set optimizer type in [SGD/Adam/AdamW...], default: AdamW')
parser.add_argument('--lr', type=float, default=3e-4,\
                    help='set learning rate, default: 3e-4')
parser.add_argument('--weightDecay', type=float, default=5e-4,\
                    help='set weight decay for L2 Regularization, default: 5e-4')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda/cpu), default: cuda')
parser.add_argument('--log_file', type=str, default='../log/log.txt',\
                    help='log file name, default: ./log/log.txt')
parser.add_argument('--savePath', type=str, default='./model/model.pt',\
                    help='path to save model, default: ./model/model.txt')
parser.add_argument('--lossRatio', type=float, default=1.,\
                    help='ratio for loss-on-rumor:loss-on-stance, default: 1.0')

def main():
    args = parser.parse_args()
    logging.basicConfig(
        filename=args.log_file, 
        filemode="w", 
        format="[%(asctime)s]:%(levelname)s: %(message)s", 
        datefmt="%d-%M-%Y %H:%M:%S", 
        level=logging.DEBUG
    )
    
    # 获取数据集及词嵌入
    print('preparing data...', end='', flush=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    dataset = RumorDataset(args.data_path, tokenizer=tokenizer)
    train_size = dataset.__len__() // 10 * 8
    dev_size = dataset.__len__() // 10 * 9 - train_size
    test_size = dataset.__len__() - train_size - dev_size
    train_set, dev_set, test_set = random_split(dataset, [train_size, dev_size, test_size])

    train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
    dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
    test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
    category = dataset.category
    print('done.', flush=True)

    # 选定实验设备
    if 'cuda' in args.device:
        device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))
        if torch.cuda.is_available():
            trainInfo = 'train MSABiGCN on device:' + args.device + '\n'
        else:
            trainInfo = 'train MSABGCN on device: cpu\n'
    else:
        device = torch.device('cpu')
        trainInfo = 'train MSABiGCN on device: cpu\n'
    logging.info(trainInfo)
    
    # 声明模型、损失函数、优化器
    bert_model = BertModel.from_pretrained(args.bert_path)

    model = MSABiGCNOnlyRumor(
        bert_model = bert_model,
        embedding_dim = args.embedding_dim,
        GCN_hidden_dim = args.GCN_hidden_dim, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumor_feature_dim = args.rumor_feature_dim, # GCN输出层的维度
        rumor_tag_num = len(category), # 谣言标签种类数
        batch_first = True,
        dropout = args.dropout, # 模型默认使用的drop out概率
        device = device
    )
    model = model.set_device(device)
    logging.info(model.__str__())
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params = model.parameters(), lr = args.lr, weight_decay = args.weightDecay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.Adam(params = model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = args.weightDecay)
    else:
        optimizer = optim.SGD(params = model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = args.weightDecay)
    
    start = 1
    # 记录验证集上的最好性能，用于early stop
    earlyStopCounter = 0
    maxF1Train = 0.
    maxF1Test = 0.

    trainRumorAcc = []
    trainRumorF1 = []
    trainLoss = []

    devRumorAcc = []
    devRumorF1 = []
    devLoss = []

    epochTime = []

    for epoch in range(start, args.epoch + 1):
        logging.info('[epoch: {:d}] '.format(epoch))

        # 训练模型
        rumorTruth = []
        rumorPre = []
        totalLoss = 0.
        
        model.train()
        startTime = process_time()
        for node_token, edge_index_TD, edge_index_BU, root_index, label in tqdm(
            train_loader, 
            desc="[epoch: {:d}] ".format(epoch), 
            leave=False, 
            ncols=100
        ):
            label = label.to(device)
            rumorTruth += label.tolist()

            optimizer.zero_grad()
            rumorPredict = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
            loss = loss_func(rumorPredict, label)
            totalLoss += loss
            loss.backward()
            optimizer.step()
            
            rumorPredict = softmax(rumorPredict, dim=1)
            rumorPre += rumorPredict.max(dim=1)[1].tolist()
        
        logging.info('average loss: {:f}\n'.format(
            totalLoss / len(train_loader)
        ))
        trainLoss.append((totalLoss / len(train_loader)).item())
        
        macroF1 = f1_score(rumorTruth, rumorPre, labels=range(len(category)), average='macro')
        acc = (np.array(rumorTruth) == np.array(rumorPre)).sum() / len(rumorTruth)
        logging,info("    rumor detection accuracy: {:.4f}, macro-f1: {:.4f}\n".format(
            acc,
            macroF1
        ))
        trainRumorAcc.append(acc)
        trainRumorF1.append(macroF1)
        maxF1Train = max(maxF1Train, macroF1)
        endTime = process_time()
        epochTime.append(endTime - startTime)
    
        # 验证集测试并保存模型
        logging.info('==================================================\ntest model on test set\n')
        rumorTruth = []
        rumorPre = []
        totalLoss = 0.

        model.eval()
        for node_token, edge_index_TD, edge_index_BU, root_index, label in tqdm(
            iter(dev_loader), 
            desc="[epoch: {:d}, test]".format(epoch), 
            leave=False, 
            ncols=80
        ):
            with torch.no_grad()
            rumorTag = thread['rumorTag'].to(device)
            rumorTruth += thread['rumorTag'].tolist()

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
            
            rumorPredict = model.forward(thread)
            loss = loss_func(rumorPredict, rumorTag)
            totalLoss += loss

            rumorPredict = softmax(rumorPredict, dim=1)
            rumorPre += rumorPredict.max(dim=1)[1].tolist()

        if datasetType == 'semEval':
            macroF1Rumor = f1_score(rumorTruth, rumorPre, labels=[0,1,2], average='macro')
        elif datasetType == 'PHEME':
            macroF1Rumor = f1_score(rumorTruth, rumorPre, labels=[0,1,2,3], average='macro')
        accRumor = (np.array(rumorTruth) == np.array(rumorPre)).sum() / len(rumorPre)
            
#==============================================
# 保存验证集marco F1和最大时的模型
            if macroF1Rumor > maxF1Test:
                model.save(args.savePath)
                earlyStopCounter = 0
                saveStatus = {
                    'epoch': epoch,
                    'macroF1Rumor': macroF1Rumor,
                    'accRumor': accRumor,
                    'loss': (totalLoss / len(testLoader)).item()
                }
                maxF1Test = max(maxF1Test, macroF1Rumor)
                f.write('saved model\n')
            else:
                earlyStopCounter += 1
#==============================================
            f.write('average joint-loss: {:f}\n'.format(totalLoss / len(testLoader)))
            f.write('rumor detection:\n')
            f.write('accuracy: {:f}, macro-f1: {:f}\n'.format(
                accRumor, 
                macroF1Rumor
            ))
            f.write('early stop counter: {:d}\n'.format(earlyStopCounter))
            f.write('========================================\n')
            devLoss.append((totalLoss / len(testLoader)).item())
            devRumorAcc.append(accRumor)
            devRumorF1.append(macroF1Rumor)
        f.close()
        if earlyStopCounter >= 10: # 验证集上连续多次测试性能没有提升就早停
            print('early stop when F1 on dev set did not increase')
            break
    print(saveStatus)
    with open(args.log_file, 'a') as f:
        f.write('\n\nsave model at epoch {:d},\
                \nmarco F1 rumor: {:f}, acc rumor {:f}\n'.format(
                    saveStatus['epoch'],
                    saveStatus['macroF1Rumor'], saveStatus['accRumor']
                ))

    saveStatus['trainRumorAcc'] = trainRumorAcc
    saveStatus['trainRumorF1'] = trainRumorF1
    saveStatus['trainLoss'] = trainLoss

    saveStatus['devRumorAcc'] = devRumorAcc
    saveStatus['devRumorF1'] = devRumorF1
    saveStatus['devLoss'] = devLoss
    saveStatus['runTime'] = epochTime
    with open(args.log_file[:-4] + '.json', 'w') as f:
        f.write(json.dumps(saveStatus))
# end main()

if __name__ == '__main__':
    main()