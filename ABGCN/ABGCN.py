import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

class ABGCN(nn.Module):
    def __init__(
        self,
        w2vDim: int, # 使用的词嵌入的维度
        s2vDim: int, # 使用的句嵌入的维度
        gcnHiddenDim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumorFeatureDim: int, # GCN输出层的维度
        numRumorTag: int, # 谣言标签种类数
        numStanceTag: int, # 立场标签种类数
        batchFirst = True,
        numHeads = 5, # multi-head attention中使用的头数
        dropout = 0.0 # 模型默认使用的drop out概率
    ):
        super().__init__()
        self.device = 'cpu'
        self.w2vDim = w2vDim
        self.s2vDim = s2vDim
        self.gcnHiddenDim = gcnHiddenDim
        self.rumorFeatureDim = rumorFeatureDim
        self.numRumorTag = numRumorTag
        self.numStanceTag = numStanceTag
        self.dropout = dropout
        self.batchFirst = batchFirst
        self.batchSize = 1 # 实际上，由于不会写支持batch化的GCN，我们把1个thread视作1个batch
        self.numHeads = numHeads

        # word2vec
        self.embed = nn.Embedding.from_pretrained()
        
        # sentence embedding
        self.wordAttention = nn.MultiheadAttention(
            embed_dim = w2vDim,
            num_heads = numHeads,
            dropout = dropout,
            batch_first = True
        )
        
        # GCN 谣言检测模块
        self.biGCN = BiGCN(self.s2vDim, self.gcnHiddenDim, self.rumorFeatureDim, self.numRumorTag)
        
        # Attention 立场分析模块
        self.stanceAttention = nn.MultiheadAttention(
            embed_dim = w2vDim,
            num_heads = numHeads,
            dropout = dropout,
            batch_first = True
        )
        #self.w2sFc = nn.Linear(w2vDim, s2vDim)
        self.stanceFc = nn.Linear(w2vDim, numStanceTag)

    def forwardRumor(self, data):
        nodeFeature = data['nodeFeature'].to(self.device) # 拷贝构造
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
        nodeFeature = data['nodeFeature'].to(self.device)
        
        nodeFeature, _ = self.wordAttention(nodeFeature, nodeFeature, nodeFeature) # self attention
        nodeFeature, _ = self.stanceAttention(nodeFeature, nodeFeature, nodeFeature)
        
        # 取出cls token对应的注意力得分
        clsAttention = []
        for post in nodeFeature:
            clsAttention.append(post[0])
        clsAttention = torch.stack(clsAttention, dim = 0)

        nodeFeature = self.stanceFc(clsAttention)

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
