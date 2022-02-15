import torch
from torch import optim
from data import *
from model import *
from random import shuffle
from sklearn.metrics import f1_score
import sys

lr = 1e-5
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
GRU_LAYER_NUM = 2
TASK1_CLASS_NUM = 3
TASK2_CLASS_NUM = 4

CUDA_DEVICE = 'cuda:1'
MAX_EPOCH = 300

def main():
    modelType = sys.argv[1] # 取得实验类型参数
    print('preparing data...', end='')
    sys.stdout.flush()
    train_set, MAX_LEN = getTrainSet()
    print('done training with model ' + modelType)
    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else 'cpu')
    model = '' #这里先声明一下模型对象
    # 选择模型
    if modelType == 'MTUS':
        model = MTUS(EMBEDDING_DIM, HIDDEN_DIM, MAX_LEN, GRU_LAYER_NUM, TASK1_CLASS_NUM, TASK2_CLASS_NUM)
        model.set_device(device)
    elif modelType == "MTES":
        model = MTES(EMBEDDING_DIM, HIDDEN_DIM, MAX_LEN, GRU_LAYER_NUM, TASK1_CLASS_NUM, TASK2_CLASS_NUM)
        model.set_device(device)
    elif modelType == 'MTUSNEW':
        model = MTUSNEW(EMBEDDING_DIM, HIDDEN_DIM, MAX_LEN, GRU_LAYER_NUM, TASK1_CLASS_NUM, TASK2_CLASS_NUM)
        model.set_device(device)
    elif modelType == "MTESNEW":
        model = MTESNEW(EMBEDDING_DIM, HIDDEN_DIM, MAX_LEN, GRU_LAYER_NUM, TASK1_CLASS_NUM, TASK2_CLASS_NUM)
        model.set_device(device)
    
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=lr) # 优化器
    
    # 划分训练集和测试集，比例约是4:1
    shuffle(train_set)
    train_set_size = len(train_set) // 5 * 4
    test_set_size = len(train_set) - train_set_size
    test_set = train_set[train_set_size:-1]
    test_set.append(train_set[len(train_set) - 1])
    train_set = train_set[0:train_set_size]
    
    best_m1_mic_f1 = 0
    best_m1_mac_f1 = 0
    best_m1_epoch = -1
    best_m2_mic_f1 = 0
    best_m2_mac_f1 = 0
    best_m2_epoch = -1
    for epoch in range(1, MAX_EPOCH + 1):
        print('[epoch {:>3d}]'.format(epoch), end=" ")
        # 打乱训练集的顺序
        shuffle(train_set)
        tags = []
        pres = []

        if epoch % 2 == 1: # 训练M1
            print('working on task 1')
            model.train()
            train_loss = 0.
            for i in range(train_set_size):
                x = train_set[i][0].to(device)
                y = train_set[i][1].to(device)
                rumor_label = train_set[i][2].to(device)
                optimizer.zero_grad()
                
                p = model.forwardM1(x).view(-1, TASK1_CLASS_NUM)
                loss = loss_func(p, rumor_label)
                loss.backward()
                optimizer.step()
                
                train_loss += loss
            train_loss = train_loss / train_set_size
            print("train set loss: {:.4f}".format(train_loss))
            
            model.eval()
            test_loss = 0.
            correct_num = 0
            for i in range(test_set_size):
                x = test_set[i][0].to(device)
                y = test_set[i][1].to(device)
                rumor_label = train_set[i][2].to(device)
                
                p = model.forwardM1(x).view(-1, TASK1_CLASS_NUM)
                loss = loss_func(p, rumor_label)
                test_loss += loss
                
                p = p.view(TASK1_CLASS_NUM)
                max_p = p.max()
                pre = p.tolist().index(max_p)
                if pre == rumor_label:
                    correct_num += 1
                tags.append(int(rumor_label[0].item()))
                pres.append(int(pre))
            test_loss = test_loss / test_set_size
            print("test set loss: {:.4f}".format(test_loss), end='. ')
            print("test set tag accuracy: {:.2f}% ({:d}/{:d})".format(100 * correct_num / test_set_size, correct_num, test_set_size))
            print('mic-F1: {:.3f}, mac-F1: {:.3f}'.format(f1_score(tags, pres, labels=[0, 1, 2], average='micro'), f1_score(tags, pres, labels=[0,1,2], average='macro')))
            if f1_score(tags, pres, labels=[0, 1, 2], average='macro') > best_m1_mac_f1:
                best_m1_mic_f1 = f1_score(tags, pres, labels=[0, 1, 2], average='micro')
                best_m1_mac_f1 = f1_score(tags, pres, labels=[0, 1, 2], average='macro')
                best_m1_epoch = epoch
        else: # 训练M2
            print('working on task 2')
            model.train()
            train_loss = 0.
            for i in range(train_set_size):
                x = train_set[i][0].to(device)
                y = train_set[i][1].to(device)
                rumor_label = train_set[i][2].to(device)
                optimizer.zero_grad()
                
                p = model.forwardM2(x)
                loss = loss_func(p, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss
            train_loss = train_loss / train_set_size
            print("train set loss: {:.4f}".format(train_loss))
            
            model.eval()
            test_loss = 0.
            correct_num = 0
            total_text = 0
            for i in range(test_set_size):
                x = test_set[i][0].to(device)
                y = test_set[i][1].to(device)
                rumor_label = train_set[i][2].to(device)
                
                ps = model.forwardM2(x)
                loss = loss_func(ps, y)
                test_loss += loss
                
                for j in range(ps.size()[0]):
                    p = ps[j]
                    max_p = p.max()
                    pre = p.tolist().index(max_p)
                    if pre == y[j]:
                        correct_num += 1
                    total_text += 1
                    tags.append(y[j].item())
                    pres.append(int(pre))
            test_loss = test_loss / test_set_size
            print("test set loss: {:.4f}".format(test_loss), end='. ')
            print("test set tag accuracy: {:.2f}% ({:d}/{:d})".format(100 * correct_num / total_text, correct_num, total_text))
            print('mic-F1: {:.3f}, mac-F1: {:.3f}'.format(f1_score(tags, pres, labels=[0, 1, 2, 3], average='micro'), f1_score(tags, pres, labels=[0,1,2,3], average='macro')))
            if f1_score(tags, pres, labels=[0,1,2,3], average='macro') > best_m2_mac_f1:
                best_m2_mic_f1 = f1_score(tags, pres, labels=[0,1,2,3], average='micro')
                best_m2_mac_f1 = f1_score(tags, pres, labels=[0,1,2,3], average='macro')
                best_m2_epoch = epoch
        print("")
    print('==========')
    print('task 1 best result(mac-F1) occour in epoch {:d}'.format(best_m1_epoch))
    print('mic-F1: {:.3f}, mac-F1: {:.3f}'.format(best_m1_mic_f1, best_m1_mac_f1))
    print('task 2 best result(mac-F1) occour in epoch {:d}'.format(best_m2_epoch))
    print('mic-F1: {:.3f}, mac-F1: {:.3f}'.format(best_m2_mic_f1, best_m2_mac_f1))

if __name__ == '__main__':
    main()