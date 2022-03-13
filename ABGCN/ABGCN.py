import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

class ABGCN(nn.Module):
    def __init__(
        self,
        word2vec, # 预训练词向量
        word2index, # 词-下表映射
        s2vDim: int, # 使用的句嵌入的维度
        gcnHiddenDim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumorFeatureDim: int, # GCN输出层的维度
        numRumorTag: int, # 谣言标签种类数
        numStanceTag: int, # 立场标签种类数
        w2vAttentionHeads = 5, # multi-head attention中使用的头数
        s2vAttentionHeads = 8,
        batchFirst = True,
        dropout = 0.0 # 模型默认使用的drop out概率
    ):
        super().__init__()
        self.device = 'cpu'
        self.s2vDim = s2vDim
        self.gcnHiddenDim = gcnHiddenDim
        self.rumorFeatureDim = rumorFeatureDim
        self.numRumorTag = numRumorTag
        self.numStanceTag = numStanceTag
        self.dropout = dropout
        self.batchFirst = batchFirst
        self.batchSize = 1 # 实际上，由于不会写支持batch化的GCN，我们把1个thread视作1个batch
        self.w2vAttentionHeads = w2vAttentionHeads
        self.s2vAttentionHeads = s2vAttentionHeads

        # 使用预训练word2vec初始化embed层的参数
        self.w2vDim = word2vec.vector_size
        weight = torch.zeros(len(word2index) + 1, self.w2vDim) # 留出0号位置给pad
        for i in range(len(word2vec.index_to_key)):
            try:
                index = word2index[word2vec.index_to_key[i]]
            except:
                continue
            weight[index] = torch.FloatTensor(word2vec[word2vec.index_to_key[i]].tolist())
        self.embed = nn.Embedding.from_pretrained(weight, freeze = False, padding_idx = 0)
        
        # sentence embed模块
        self.wordAttention = nn.MultiheadAttention(
            embed_dim = self.w2vDim,
            num_heads = w2vAttentionHeads,
            dropout = dropout,
            batch_first = True
        )
        self.s2vFc = nn.Linear(self.w2vDim, self.s2vDim)
        
        # GCN 谣言检测模块
        self.biGCN = BiGCN(self.s2vDim, self.gcnHiddenDim, self.rumorFeatureDim, self.numRumorTag)
        self.RumorFc = nn.Linear((rumorFeatureDim + gcnHiddenDim) * 2, numRumorTag)

        # Attention立场分析模块
        self.stanceAttention = nn.MultiheadAttention(
            embed_dim = self.s2vDim,
            num_heads = s2vAttentionHeads,
            dropout = dropout,
            batch_first = True
        )
        self.stanceFc = nn.Linear(self.s2vDim, numStanceTag)

    # 根据输入的任务标识进行前向迭代，
    def forward(self, thread, mission: int):
        shape = list(thread['nodeText'].shape)
        shape.append(-1)
        nodeText = thread['nodeText'].view(-1).to(self.device)
        nodeFeature = self.embed(nodeText).view(tuple(shape))
        nodeFeature, _ = self.wordAttention(nodeFeature, nodeFeature, nodeFeature)
        nodeFeature = torch.tanh(self.s2vFc(nodeFeature))
        
        if(mission == 1):
            # 取出<start> token对应的Attention得分作为节点的sentence Embedding
            s2v = []
            for post in nodeFeature:
                s2v.append(post[0])
            s2v = torch.stack(s2v, dim = 0)
            dataTD = Data(
                x = torch.clone(s2v).to(self.device), 
                edgeIndex = thread['edgeIndexTD'].to(self.device), 
                rootIndex = thread['threadIndex']
            )
            dataBU = Data(
                x = torch.clone(s2v).to(self.device), 
                edgeIndex = thread['edgeIndexBU'].to(self.device), 
                rootIndex = thread['threadIndex']
            )
            nodeFeature = self.biGCN(dataTD, dataBU)
            return self.RumorFc(nodeFeature)
        else:
            nodeFeature, _ = self.stanceAttention(nodeFeature, nodeFeature, nodeFeature)
            # 取出<start> token对应的Attention得分作为节点的stance特征
            stanceFeature = []
            for post in nodeFeature:
                stanceFeature.append(post[0])
            stanceFeature = torch.stack(stanceFeature, dim = 0)
            return self.stanceFc(stanceFeature)

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