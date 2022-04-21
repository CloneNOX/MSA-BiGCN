import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping3Class
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
from time import process_time
import copy
import json

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x=F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)
        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=th.nn.Linear((out_feats+hid_feats)*2,3)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

def train_GCN(treeDic, x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,epochTime):
    log = open('./SemEvalLog.txt', 'a')

    model = Net(5000,64,64).to(device)
    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    train_losses,val_losses,train_accs,val_accs = [],[],[],[]
    early_stopping = EarlyStopping3Class(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize,
                                  shuffle=False, num_workers=10)
        test_loader = DataLoader(testdata_list, batch_size=batchsize,
                                 shuffle=True, num_workers=10)
        avg_loss,avg_acc = [],[]
        batch_idx = 0
        model.train()
        tqdm_train_loader = tqdm(train_loader)
        start = process_time()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels = model(Batch_data)
            # print(out_labels, Batch_data.y)
            loss = F.nll_loss(out_labels, Batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            postfix = "Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,
                                                                                                                   epoch,
                                                                                                                   batch_idx,
                                                                                                                   loss.item(),
                                                                                                                   train_acc)
            tqdm_train_loader.set_postfix_str(postfix)
            batch_idx = batch_idx + 1
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        end = process_time()
        epochTime.append(end - start)

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3 = [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc3, Prec3, Recll3, F3 = evaluation3class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1)
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(
            epoch, np.mean(temp_val_losses), np.mean(temp_val_accs)),
            file=log
        )

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
        ]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), model, 'BiGCN', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            break
    log.close()
    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3

log = open('./SemEvalLog.txt', 'w')
log.close()
lr=0.0005
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=16
tddroprate=0
budroprate=0
datasetname="SemEval"
iterations=int(sys.argv[1])
model="BiGCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs,TR_F1,FR_F1,UR_F1  = [],[],[],[]
epochTime = []
for iter in range(iterations):
    fold0_x_test, fold0_x_train,\
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train,  \
    fold3_x_test, fold3_x_train,  \
    fold4_x_test, fold4_x_train = load5foldData(datasetname) # 交叉验证数据集
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs, accs_0, F1_0, F2_0, F3_0 = train_GCN(treeDic,
                                                                                        fold0_x_test,
                                                                                        fold0_x_train,
                                                                                        tddroprate,budroprate,
                                                                                        lr, weight_decay,
                                                                                        patience,
                                                                                        n_epochs,
                                                                                        batchsize,
                                                                                        datasetname,
                                                                                        iter,
                                                                                        epochTime)
    train_losses, val_losses, train_accs, val_accs, accs_1, F1_1, F2_1, F3_1 = train_GCN(treeDic,
                                                                                        fold1_x_test,
                                                                                        fold1_x_train,
                                                                                        tddroprate,budroprate, lr,
                                                                                        weight_decay,
                                                                                        patience,
                                                                                        n_epochs,
                                                                                        batchsize,
                                                                                        datasetname,
                                                                                        iter,
                                                                                        epochTime)
    train_losses, val_losses, train_accs, val_accs, accs_2, F1_2, F2_2, F3_2 = train_GCN(treeDic,
                                                                                        fold2_x_test,
                                                                                        fold2_x_train,
                                                                                        tddroprate,budroprate, lr,
                                                                                        weight_decay,
                                                                                        patience,
                                                                                        n_epochs,
                                                                                        batchsize,
                                                                                        datasetname,
                                                                                        iter,
                                                                                        epochTime)
    train_losses, val_losses, train_accs, val_accs, accs_3, F1_3, F2_3, F3_3 = train_GCN(treeDic,
                                                                                        fold3_x_test,
                                                                                        fold3_x_train,
                                                                                        tddroprate,budroprate, lr,
                                                                                        weight_decay,
                                                                                        patience,
                                                                                        n_epochs,
                                                                                        batchsize,
                                                                                        datasetname,
                                                                                        iter,
                                                                                        epochTime)
    train_losses, val_losses, train_accs, val_accs, accs_4, F1_4, F2_4, F3_4 = train_GCN(treeDic,
                                                                                        fold4_x_test,
                                                                                        fold4_x_train,
                                                                                        tddroprate,budroprate, lr,
                                                                                        weight_decay,
                                                                                        patience,
                                                                                        n_epochs,
                                                                                        batchsize,
                                                                                        datasetname,
                                                                                        iter,
                                                                                        epochTime)
    test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    UR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
print("Total_Test_Accuracy: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))

saveStatus = {}
saveStatus['runTime'] = epochTime
with open('SemevalTrainTime.json', 'w') as f:
    f.write(json.dumps(saveStatus))

log = open('./SemEvalLog.txt', 'a')
print("Total_Test_Accuracy: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations),
    file=log
)

log.close()
