import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

class MSABiGCNOnlyRumor(nn.Module):
    def __init__(
        self,
        word2vec, # 预训练词向量
        word2index, # 词-下表映射
        s2vDim: int, # 使用的句嵌入的维度
        gcnHiddenDim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumorFeatureDim: int, # GCN输出层的维度
        numRumorTag: int, # 谣言标签种类数
        w2vAttentionHeads = 5, # word2vec multi-head attention中使用的头数
        s2vAttentionHeads = 8, # sentence2vec multi-head attention中使用的头数
        batchFirst = True,
        dropout = 0.0 # 模型默认使用的drop out概率
    ):
        super().__init__()
        self.device = 'cpu'
        self.s2vDim = s2vDim
        self.gcnHiddenDim = gcnHiddenDim
        self.rumorFeatureDim = rumorFeatureDim
        self.numRumorTag = numRumorTag
        self.batchFirst = batchFirst
        self.batchSize = 1 # 实际上，由于不会写支持batch化的GCN，我们把1个thread视作1个batch
        self.w2vAttentionHeads = w2vAttentionHeads
        self.s2vAttentionHeads = s2vAttentionHeads
        self.dropout = dropout

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
        self.s2vRumor = nn.Linear(self.w2vDim, self.s2vDim)
        
        # GCN 谣言检测模块
        self.biGCN = BiGCN(
            self.s2vDim, 
            self.gcnHiddenDim, 
            self.rumorFeatureDim, 
            self.numRumorTag, 
            numLayers = 2,
            dropout = dropout)
        self.f2tRumor = nn.Linear((rumorFeatureDim + gcnHiddenDim) * 2, numRumorTag)

    # 根据输入的任务标识进行前向迭代，
    def forward(self, thread):
        shape = list(thread['nodeText'].shape)
        shape.append(-1)
        nodeText = thread['nodeText'].view(-1).to(self.device)
        nodeFeature = self.embed(nodeText).view(tuple(shape))
        nodeFeature, _ = self.wordAttention(nodeFeature, nodeFeature, nodeFeature)
        
        # rumor detection:
        # 取出<start> token对应的Attention得分作为节点的sentence Embedding
        s2v = []
        for post in nodeFeature:
            s2v.append(post[0])
        s2v = torch.stack(s2v, dim = 0)
        s2v = torch.tanh(self.s2vRumor(s2v))
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
        rumorFeature = self.biGCN(dataTD, dataBU)

        return self.f2tRumor(rumorFeature)

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

class MSAOnlyStance(nn.Module):
    def __init__(
        self,
        word2vec, # 预训练词向量
        word2index, # 词-下表映射
        s2vDim: int, # 使用的句嵌入的维度
        numStanceTag: int, # 立场标签种类数
        w2vAttentionHeads = 5, # word2vec multi-head attention中使用的头数
        s2vAttentionHeads = 8, # sentence2vec multi-head attention中使用的头数
        batchFirst = True,
        dropout = 0.0 # 模型默认使用的drop out概率
    ):
        super().__init__()
        self.device = 'cpu'
        self.s2vDim = s2vDim
        self.numStanceTag = numStanceTag
        self.batchFirst = batchFirst
        self.batchSize = 1 # 实际上，由于不会写支持batch化的GCN，我们把1个thread视作1个batch
        self.w2vAttentionHeads = w2vAttentionHeads
        self.s2vAttentionHeads = s2vAttentionHeads
        self.dropout = dropout

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
        self.s2vStance = nn.Linear(self.w2vDim, self.s2vDim)

        # Attention立场分析模块
        self.stanceAttention = nn.MultiheadAttention(
            embed_dim = self.s2vDim,
            num_heads = s2vAttentionHeads,
            dropout = dropout,
            batch_first = True
        )
        self.stanceFc = nn.Linear(self.s2vDim, self.s2vDim)
        self.f2tStance = nn.Linear(self.s2vDim, numStanceTag)

    # 根据输入的任务标识进行前向迭代，
    def forward(self, thread):
        shape = list(thread['nodeText'].shape)
        shape.append(-1)
        nodeText = thread['nodeText'].view(-1).to(self.device)
        nodeFeature = self.embed(nodeText).view(tuple(shape))
        nodeFeature, _ = self.wordAttention(nodeFeature, nodeFeature, nodeFeature)
        
        # stance classification: 取出<start> token对应的Attention得分作为节点的stance特征
        stanceFeatureAll = torch.tanh(self.s2vStance(nodeFeature))
        stanceFeatureAll, _ = self.stanceAttention(stanceFeatureAll, stanceFeatureAll, stanceFeatureAll)
        stanceFeatureAll = torch.tanh(self.stanceFc(stanceFeatureAll))
        stanceFeature = []
        for post in stanceFeatureAll:
            stanceFeature.append(post[0])
        stanceFeature = torch.stack(stanceFeature, dim = 0)

        return self.f2tStance(stanceFeature)

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
    def __init__(self, inputDim, hiddenDim, outDim, numLayers = 2, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(inputDim, hiddenDim)
        self.conv2 = GCNConv(hiddenDim + inputDim, outDim)
        self.dropout = dropout

    def forward(self, data):
        posts, edge_index, rootIndex = data.x, data.edgeIndex, data.rootIndex # posts(n, inputDim), edgeIndex(2, |E|)
        
        postRoot = torch.clone(posts[rootIndex])
        postRoot = postRoot.repeat(posts.shape[0], 1)
        conv1Out = self.conv1(posts, edge_index)
        conv1Root = conv1Out[rootIndex]

        conv2In = torch.cat([postRoot, conv1Out], dim=1)
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
    def __init__(self, inputDim, hiddenDim, convOutDim, NumRumorTag, numLayers = 2, dropout = 0.5):
        super(BiGCN, self).__init__()
        self.TDGCN = GCN(inputDim, hiddenDim, convOutDim, numLayers, dropout)
        self.BUGCN = GCN(inputDim, hiddenDim, convOutDim, numLayers, dropout)
        
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