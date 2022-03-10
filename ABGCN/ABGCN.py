import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class ABGCN(nn.Module):
    def __init__(
        self,
        w2vDim: int, # 使用的词嵌入的维度
        s2vDim: int, # 使用的句嵌入的维度
        gcnHiddenDim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumorFeatureDim: int, # GCN输出层的维度
        numRumorTag: int, # 谣言标签种类数
        numStanceTag: int, # 立场标签种类数
        s2vMethon = 'a', # 获取据嵌入的方法（l:lstm; a:attention）
        batchFirst = True,
        # ======
        # 仅在s2vMethod == 'a'时有效
        numHeads = 8, # multi-head attention中使用的头数
        numWordAttentionLayer = 2, # 获取句嵌入的self attention模块层数
        numStanceAttentionLayer = 2, # 获取StanceTag的self attention模块层数
        # ======
        # 仅在s2vMethod == 'l'时有效
        numLstmLayer = 1, # lstm层数
        # ======
        dropout = 0.0 # 模型默认使用的drop out概率
    ):
        super().__init__()
        self.device = 'cpu'
        self.w2vDim = w2vDim
        self.s2vDim = s2vDim
        self.gcnHiddenDim = gcnHiddenDim
        self.rumorFeatureDim = rumorFeatureDim
        self.s2vMethon = s2vMethon
        self.numRumorTag = numRumorTag
        self.numStanceTag = numStanceTag
        self.dropout = dropout
        self.batchFirst = batchFirst
        self.batchSize = 1 # 实际上，由于不会写支持batch化的GCN，我们把1个thread视作1个batch

        # sentence embedding
        # ==使用biLSTM获取post的向量表示==
        if self.s2vMethon == 'l':
            self.numLstmLayer = numLstmLayer
            self.lstm = nn.LSTM(
                input_size = self.w2vDim,
                hidden_size = self.s2vDim,
                num_layers = self.numLstmLayer,
                bidirectional=True
            )
            self.h0 = nn.Parameter(
                torch.randn((self.numLstmLayer * 2, self.batchSize, self.s2vDim))
            )
            self.c0 = nn.Parameter(
                torch.randn((self.numLstmLayer * 2, self.batchSize, self.s2vDim))
            )
            self.s2vDim *= 2 # 由于使用BiDirect所以s2vDim的维度会扩大1倍
        # ==使用Attention==
        else:
            self.numHeads = numHeads
            self.numWordAttentionLayer = numWordAttentionLayer
            self.wordAttention = SelfAttention(
                embed_dim = w2vDim,
                numHeads = numHeads,
                numLayer = numWordAttentionLayer,
                dropout = dropout,
                batchFirst = batchFirst
            )
        # 在经过wordAttention后得到的是每个post内词层面的Attention, 要用在谣言检测
        
        # GCN 谣言检测模块
        self.biGCN = BiGCN(self.s2vDim, self.gcnHiddenDim, self.rumorFeatureDim, self.numRumorTag)
        # Attention 立场分析模块
        self.stanceAttention = SelfAttention(
            embedDim = w2vDim,
            numHeads = numHeads,
            numLayer = numStanceAttentionLayer,
            dropout = dropout,
            batchFirst = batchFirst
        )
        self.w2sFc = nn.Linear(w2vDim, s2vDim)
        self.stanceFc = nn.Linear(s2vDim, numStanceTag)
 
    def forwardS2V(self, nodeFeature):
        # 把w2v转化成s2v作为节点的特征
        if self.s2vMethon == 'l': # 使用RNN的方法，取序列的最后一个隐状态作为句子的向量表示
            s2v = []
            # for w2v in nodeFeature:
            #     w2v = w2v.view((len(w2v), 1, -1)).to(self.device)
            #     sentenceHidden, _ = self.lstm(w2v, (self.h0, self.c0))
            #     # 仅取出最后一层的隐状态作为s2v
            #     s2v.append(sentenceHidden[-1].view(1, len(w2v[-1]), -1))
            # s2v = torch.cat(s2v, dim=0)

        else: # 使用Attention的方法，为了后续分任务学习，先不把word粒度的Attention进行聚合
            nodeFeature = self.wordAttention(nodeFeature) # nodeFeature.shape = (batch, len, w2vDim)
        return nodeFeature

    def forwardRumor(self, data):
        nodeFeature = torch.Tensor(data['nodeFeature']).to(self.device) # 拷贝构造
        self.forwardS2V(nodeFeature)
        # 把所有词的Attention表示平均作为句子的向量表示(待定，需要比较采取开头/结尾的表示的方法)
        nodeFeature = torch.mean(nodeFeature, dim = 1)

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

    def forwardStance(self, data):
        nodeFeature = torch.Tensor(data['nodeFeature']).to(self.device) # 拷贝构造
        self.forwardS2V(nodeFeature)

        # 保持word粒度的Attention不变，接入到stanceAttention
        nodeFeature = self.stanceAttention(nodeFeature)
        nodeFeature = self.w2sFc(nodeFeature)
        # 把所有词的Attention表示平均作为句子的向量表示(待定，需要比较采取开头/结尾的表示的方法)
        nodeFeature = F.relu(torch.mean(nodeFeature, dim = 1))
        nodeFeature = self.stanceFc(nodeFeature)
        return nodeFeature

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

# GCN实现
class GCN(torch.nn.Module):
    def __init__(self, inputDim, hiddenDim, outDim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(inputDim, hiddenDim)
        self.conv2 = GCNConv(hiddenDim + inputDim, outDim)
        self.dropout = dropout

    def forward(self, data):
        posts, edge_index, rootIndex = data.x, data.edgeIndex, data.rootIndex # posts(n, inputDim), edgeIndex(2, |E|)
        
        conv1Out = self.conv1(posts, edge_index)
        postRoot = torch.clone(posts[rootIndex])
        postRoot = postRoot.repeat(posts.shape[0], 1)
        conv1Root = conv1Out[rootIndex]

        conv2In = torch.cat([conv1Out, postRoot], dim=1)
        conv2In = F.relu(conv2In)
        conv2In = F.dropout(conv2In, training=self.training, p=self.dropout) # BiGCN对于dropout的实现，一次卷积之后随机舍弃一些点
        conv2Out = self.conv2(conv2In, edge_index)
        conv2Out = F.relu(conv2Out)

        conv1Root = conv1Root.repeat(posts.shape[0], 1)
        feature = torch.cat([conv1Root, conv2Out], dim=1)
        # 使用均值计算，把所有节点的特征聚合成为图的特征
        feature = torch.mean(feature, dim=0).view(1, -1)
        return feature
    
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

# BiGCN
class BiGCN(torch.nn.Module):
    def __init__(self, inputDim, hiddenDim, convOutDim, NumRumorTag):
        super(BiGCN, self).__init__()
        self.TDGCN = GCN(inputDim, hiddenDim, convOutDim)
        self.BUGCN = GCN(inputDim, hiddenDim, convOutDim)
        self.fc=torch.nn.Linear((convOutDim + hiddenDim) * 2, NumRumorTag)

    def forward(self, dataTD, dataBU):
        TDOut = self.TDGCN(dataTD)
        BUOut = self.BUGCN(dataBU)
        feature = torch.cat((TDOut, BUOut), dim=1)
        p = self.fc(feature)
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

# 残差self attention层
class SelfAttention(torch.nn.Module):
    def __init__(self, embedDim, numHeads, numLayer, dropout, batchFirst) -> None:
        super().__init__()
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.numLayer = numLayer
        self.dropout = dropout

        # 多层残差self attention
        self.resAtt = [nn.MultiheadAttention(
            embed_dim = embedDim,
            num_head = numHeads,
            dropout = dropout,
            batch_first = batchFirst
        ) for _ in range(numLayer)]

    def forward(self, input):
        for att in self.resAtt:
            output = att(input)
        return output