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
import json
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
parser.add_argument('--data_type', type=str, choices=['rumor','rumor-stance'], default='rumor',\
                    help="dataset's type, choice in [rumor, rumor-stance], default: rumor")
# train parameters
parser.add_argument('--optimizer', type=str, default='AdamW',\
                    help='set optimizer type in [SGD/Adam/AdamW...], default: AdamW')
parser.add_argument('--lr', type=float, default=1e-4,\
                    help='set learning rate, default: 1e-4')
parser.add_argument('--weightDecay', type=float, default=5e-4,\
                    help='set weight decay for L2 Regularization, default: 5e-4')
parser.add_argument('--epoch', type=int, default=100,\
                    help='epoch to train, default: 100')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda/cpu), default: cuda')
parser.add_argument('--log_file', type=str, default='../log/log.txt',\
                    help='log file name, default: ./log/log.txt')
parser.add_argument('--savePath', type=str, default='./model/best.pt',\
                    help='path to save model, default: ./model/best.pt')
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

    if args.data_type == 'rumor':
        dataset = RumorDataset(args.data_path, tokenizer=tokenizer)
        train_size = dataset.__len__() // 10 * 8
        dev_size = dataset.__len__() // 10 * 9 - train_size
        test_size = dataset.__len__() - train_size - dev_size
        train_set, dev_set, test_set = random_split(dataset, [train_size, dev_size, test_size])

        train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        category = dataset.category
    elif args.data_type == 'rumor-stance':
        pass
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
    bert_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params, model.parameters())
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW([
            {'params': base_params}
        ], lr = args.lr, weight_decay = args.weightDecay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.Adam([
            {'params': base_params}
        ], lr = args.lr, weight_decay = args.weightDecay)
    else:
        optimizer = optim.SGD([
            {'params': base_params}
        ], lr = args.lr, momentum = 0.9, weight_decay = args.weightDecay)
    
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
            desc="[epoch: {:d}, training] ".format(epoch), 
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
        logging.info("    rumor detection accuracy: {:.4f}, macro-f1: {:.4f}".format(
            acc,
            macroF1
        ))
        trainRumorAcc.append(acc)
        trainRumorF1.append(macroF1)
        maxF1Train = max(maxF1Train, macroF1)
        endTime = process_time()
        epochTime.append(endTime - startTime)
    
        # 验证集测试并保存模型
        logging.info('==================================================')
        logging.info('test model on dev set')
        rumorTruth = []
        rumorPre = []
        totalLoss = 0.

        model.eval()
        for node_token, edge_index_TD, edge_index_BU, root_index, label in tqdm(
            dev_loader, 
            desc="[epoch: {:d}, dev set test]".format(epoch), 
            leave=False, 
            ncols=80
        ):
            with torch.no_grad():
                rumorTag = label.to(device)
                rumorTruth += label.tolist()

                rumorPredict = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
                loss = loss_func(rumorPredict, rumorTag)
                totalLoss += loss

                rumorPredict = softmax(rumorPredict, dim=1)
            rumorPre += rumorPredict.max(dim=1)[1].tolist()

        macroF1 = f1_score(rumorTruth, rumorPre, labels=range(len(category)), average='macro')
        accRumor = (np.array(rumorTruth) == np.array(rumorPre)).sum() / len(rumorPre)
            
        #==============================================
        # 保存验证集marco F1和最大时的模型
        if macroF1 > maxF1Test:
            model.save(args.savePath)
            earlyStopCounter = 0
            saveStatus = {
                'epoch': epoch,
                'macroF1Rumor': macroF1,
                'accRumor': accRumor,
                'loss': (totalLoss / len(dev_loader)).item()
            }
            maxF1Test = max(maxF1Test, maxF1Test)
            logging.info('saved model')
        else:
            earlyStopCounter += 1
        #==============================================
        logging.info('average joint-loss: {:f}'.format(totalLoss / len(dev_loader)))
        logging.info('rumor detection:')
        logging.info('accuracy: {:f}, macro-f1: {:f}'.format(
            accRumor, 
            macroF1
        ))
        logging.info('early stop counter: {:d}\n'.format(earlyStopCounter))
        logging.info('========================================')
        devLoss.append((totalLoss / len(dev_loader)).item())
        devRumorAcc.append(accRumor)
        devRumorF1.append(macroF1)
        if earlyStopCounter >= 50: # 验证集上连续多次测试性能没有提升就早停
            logging.info('early stop when F1 on dev set did not increase')
            break
    logging.info(saveStatus)
    logging.info('\n\nbest model save at epoch {:d},\
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
    logging.info(json.dumps(saveStatus))
# end main()

if __name__ == '__main__':
    main()