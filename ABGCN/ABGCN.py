import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from biGCN import *

class ABGCN(nn.Module):
    def __init__(self,
                 w2vDim: int, # 使用的词嵌入的维度
                 s2vDim: int, # 使用的句嵌入的维度
                 gcnHiddenDim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
                 rumorFeatureDim: int, # GCN输出层的维度
                 numRumorTag: int, # 谣言标签种类数
                 numStanceTag: int, # 立场标签种类数
                 s2vMethon = 'a', # 获取据嵌入的方法（l:lstm; a:attention）
                 numLstmLayer = 1 # lstm层数，仅在s2vMethod == 'l'时有效
                ):
        super().__init__()

        self.w2vDim = w2vDim
        self.s2vDim = s2vDim
        self.gcnHiddenDim = gcnHiddenDim
        self.rumorFeatureDim = rumorFeatureDim
        self.s2vMethon = s2vMethon
        self.batchSize = 1 # 实际上，由于不会写支持batch化的GCN，模型时不支持batch化训练的
        self.numRumorTag = numRumorTag
        self.numStanceTag = numStanceTag
        self.device = 'cpu'

        # ==使用biLSTM获取post的向量表示==
        if self.s2vMethon == 'l':
            self.numLstmLayer = numLstmLayer
            self.lstm = nn.LSTM(input_size = self.w2vDim,
                                hidden_size = self.s2vDim,
                                num_layers = self.numLstmLayer,
                                bidirectional=True)
            self.h0 = nn.Parameter(torch.randn((self.numLstmLayer * 2, self.batchSize, self.s2vDim)))
            self.c0 = nn.Parameter(torch.randn((self.numLstmLayer * 2, self.batchSize, self.s2vDim)))
            self.s2vDim *= 2 # 由于使用BiDirect所以s2vDim的维度会扩大1倍
        # ==使用Attention==
        else:
            pass
        
        self.biGCN = BiGCN(self.s2vDim, self.gcnHiddenDim, self.rumorFeatureDim, self.numRumorTag)

    def forwardRumor(self, data):
        # 各节点的特征表示，此时是word2vec的形式
        nodeFeature = data['nodeFeature']

        # 把w2v转化成s2v作为节点的特征
        s2v = []
        if self.s2vMethon == 'l':
            for w2v in nodeFeature:
                w2v = w2v.view((len(w2v), 1, -1)).to(self.device)
                sentenceHidden, _ = self.lstm(w2v, (self.h0, self.c0))
                # 仅取出最后一层的隐状态作为s2v
                s2v.append(sentenceHidden[-1].view(1, len(w2v[-1]), -1))
            s2v = torch.cat(s2v, dim=0)
        else:
            pass

        # GCN处理
        s2v = s2v.view(s2v.shape[0], -1)
        dataTD = Data(x = s2v.to(self.device), 
                      edgeIndex = data['edgeIndexTD'].to(self.device), 
                      rootIndex = data['threadIndex'])
        dataBU = Data(x = s2v.to(self.device), 
                      edgeIndex = data['edgeIndexBU'].to(self.device), 
                      rootIndex = data['threadIndex'])
        p = self.biGCN(dataTD, dataBU).view(self.batchSize, -1) # p.shape = (1, *)
        
        return p
        

    # 更换计算设备
    def set_device(self, device: torch.device) -> torch.nn.Module:
        _model = self.to(device)
        _model.device = device
        return _model
    # 保存模型
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    # 加载模型
    def load(self, path: str):
        self.load_state_dict(torch.load(path))