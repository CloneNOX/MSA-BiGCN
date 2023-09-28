import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
from BiGCN import BiGCN

class BertBiGCNOnlyRumor(nn.Module):
    def __init__(
        self,
        bert_model: BertModel, 
        embedding_dim: int,
        GCN_hidden_dim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumor_feature_dim: int, # GCN输出层的维度
        rumor_tag_num: int, # 谣言标签种类数
        batch_first = True,
        dropout = 0.0, # 模型默认使用的drop out概率
        edge_drop_rate = 0.1,
        device = torch.device('cuda')
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.GCN_hidden_dim = GCN_hidden_dim
        self.rumor_feature_dim = rumor_feature_dim
        self.rumor_tag_num = rumor_tag_num
        self.batch_first = batch_first
        self.dropout = dropout
        self.edge_drop_rate = edge_drop_rate

        self.bert_model = bert_model.to(self.device)
        
        # GCN 谣言检测模块
        self.biGCN = BiGCN(
            self.embedding_dim, 
            self.GCN_hidden_dim, 
            self.rumor_feature_dim, 
            self.rumor_tag_num, 
            dropout = dropout
        ).to(self.device)
        self.fc = nn.Linear((rumor_feature_dim + GCN_hidden_dim) * 2, (rumor_feature_dim + GCN_hidden_dim)).to(self.device)
        self.feature2label = nn.Linear((rumor_feature_dim + GCN_hidden_dim), rumor_tag_num).to(self.device)

    def forward(self, node_token, edge_index_TD, edge_index_BU, root_index):        
        '''
        Iutput: 
            - node_token: transformers.BertTokenizer类的输出, 一般是, 未转成句向量
            - edge_index_TD: [[int]. [int]], [2, |E|], 自顶向下的传播树图
            - edge_index_BU: [[int]. [int]], [2, |E|], 自底向上的传播树图
            - root_index: int
        '''
        # 使用bert生成节点特征向量
        for key in node_token:
            node_token[key] = node_token[key].to(self.device)
        encode_output = self.bert_model(**node_token)
        word_embedding = encode_output.last_hidden_state
        sentence_embedding = encode_output.pooler_output

        # 采样生成保留边的编号，注意模型在eval模式下不应该dropedge
        if self.training:
            edgeNum = len(edge_index_TD[0])
            savedIndex = sample(range(edgeNum), int(edgeNum * (1 - self.edge_drop_rate)))
            savedIndex = sorted(savedIndex)
            edgeTD = [[edge_index_TD[row][col] for col in savedIndex] for row in range(2)]
            edgeTD = torch.LongTensor(edgeTD)
        else:
            edgeTD = torch.LongTensor(edge_index_TD)
        dataTD = Data(
            x = torch.clone(sentence_embedding).to(self.device), 
            edgeIndex = edgeTD.to(self.device), 
            rootIndex = root_index
        )
        if self.training:
            edgeNum = len(edge_index_BU[0])
            savedIndex = sample(range(edgeNum), int(edgeNum * (1 - self.edge_drop_rate)))
            savedIndex = sorted(savedIndex)
            edgeBU = [[edge_index_BU[row][col] for col in savedIndex] for row in range(2)]
            edgeBU = torch.LongTensor(edgeBU)
        else:
            edgeBU = torch.LongTensor(edge_index_BU)
        dataBU = Data(
            x = torch.clone(sentence_embedding).to(self.device), 
            edgeIndex = edgeBU.to(self.device), 
            rootIndex = root_index
        )
        rumorFeature = self.biGCN(dataTD, dataBU)

        rumorFeature = torch.tanh(self.fc(rumorFeature))
        return self.feature2label(rumorFeature)

    def set_device(self, device: torch.device) -> torch.nn.Module:
        '''
        更换计算设备
        '''
        _model = self.to(device)
        _model.device = device
        return _model
    
    def save(self, path: str):
        '''
        保存模型
        '''
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        '''
        加载模型
        '''
        self.load_state_dict(torch.load(path))

class BertMSA(nn.Module):
    def __init__(
        self,
        bert_model: BertModel, 
        embedding_dim: int,
        num_stance_tag: int, # 立场标签种类数
        s2v_attention_heads = 8, # sentence2vec multi-head attention中使用的头数
        batch_first = True,
        dropout = 0.0, # 模型默认使用的drop out概率
        device = torch.device('cuda')
    ):
        super().__init__()
        self.bert_model = bert_model
        self.embedding_dim = embedding_dim
        self.num_stance_tag = num_stance_tag
        self.batch_first = batch_first
        self.s2v_attention_heads = s2v_attention_heads
        self.dropout = dropout
        self.device = device

        self.bert_model = bert_model.to(self.device)

        # Attention立场分析模块
        self.s2vStance = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.stanceAttention = nn.MultiheadAttention(
            embed_dim = self.embedding_dim,
            num_heads = s2v_attention_heads,
            dropout = dropout,
            batch_first = True
        ).to(self.device)
        self.feature2label = nn.Linear(self.embedding_dim, num_stance_tag).to(self.device)

    # 根据输入的任务标识进行前向迭代，
    def forward(self, node_token):
        '''
        Iutput: 
            - node_token: transformers.BertTokenizer类的输出, 一般是, 未转成句向量
        '''
        # 使用bert生成节点特征向量
        for key in node_token:
            node_token[key] = node_token[key].to(self.device)
        encode_output = self.bert_model(**node_token)
        word_embedding = encode_output.last_hidden_state
        sentence_embedding = encode_output.pooler_output
        
        # stance classification: 取出<start> token对应的Attention得分作为节点的stance特征
        stanceFeatureAll = torch.tanh(self.s2vStance(word_embedding.to(self.device)))
        stanceFeatureAll, _ = self.stanceAttention(stanceFeatureAll, stanceFeatureAll, stanceFeatureAll)
        stanceFeature = []
        for post in stanceFeatureAll:
            stanceFeature.append(post[0])
        stanceFeature = torch.stack(stanceFeature, dim = 0)

        return self.feature2label(stanceFeature)

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

class BertBiGCN(nn.Module):
    def __init__(
        self,
        bert_model: BertModel, 
        embedding_dim: int,
        GCN_hidden_dim: int, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumor_feature_dim: int, # GCN输出层的维度
        rumor_tag_num: int, # 谣言标签种类数
        num_stance_tag: int, # 立场标签种类数
        s2v_attention_heads = 8, # sentence2vec multi-head attention中使用的头数
        batch_first = True,
        need_stance = False,
        dropout = 0.0, # 模型默认使用的drop out概率
        edge_drop_rate = 0.1,
        device = torch.device('cuda')
    ):
        super().__init__()
        self.device = device
        self.GCN_hidden_dim = GCN_hidden_dim
        self.rumor_feature_dim = rumor_feature_dim
        self.rumor_tag_num = rumor_tag_num
        self.num_stanc_tag = num_stance_tag
        self.batch_first = batch_first
        self.s2v_attention_heads = s2v_attention_heads
        self.needStance = need_stance
        self.dropout = dropout
        self.edge_drop_rate = edge_drop_rate

        self.bert_model = bert_model.to(self.device)
        
        # sentence embed模块
        self.wordAttention = nn.MultiheadAttention(
            embed_dim = self.w2vDim,
            num_heads = w2vAttentionHeads,
            dropout = dropout,
            batch_first = True
        )
        self.s2vRumor = nn.Linear(self.w2vDim, self.s2vDim)
        self.s2vStance = nn.Linear(self.w2vDim, self.s2vDim)
        
        # GCN 谣言检测模块
        if needStance:
            self.biGCN = BiGCN(
                2 * self.s2vDim, 
                self.GCN_hidden_dim, 
                self.rumor_feature_dim, 
                self.rumor_tag_num, 
                numLayers = 2,
                dropout = dropout)
        else:
            self.biGCN = BiGCN(
                self.s2vDim, 
                self.GCN_hidden_dim, 
                self.rumor_feature_dim, 
                self.rumor_tag_num, 
                numLayers = 2,
                dropout = dropout)
        self.f2tRumor = nn.Linear((rumor_feature_dim + GCN_hidden_dim) * 2, rumor_tag_num)

        # Attention立场分析模块
        self.stanceAttention = nn.MultiheadAttention(
            embed_dim = self.s2vDim,
            num_heads = s2vAttentionHeads,
            dropout = dropout,
            batch_first = True
        )
        self.stanceFc = nn.Linear(self.s2vDim, self.s2vDim)
        self.f2tStance = nn.Linear(self.s2vDim, numStanceTag)
        
        self.s2vStance = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.stanceAttention = nn.MultiheadAttention(
            embed_dim = self.embedding_dim,
            num_heads = s2v_attention_heads,
            dropout = dropout,
            batch_first = True
        ).to(self.device)

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
        
        # rumor detection:
        # 取出<start> token对应的Attention得分作为节点的sentence Embedding
        s2v = []
        for post in nodeFeature:
            s2v.append(post[0])
        s2v = torch.stack(s2v, dim = 0)
        s2v = torch.tanh(self.s2vRumor(s2v))
        # rumor detection的预测需要结合stance feature
        if self.needStance:
            s2v = torch.cat([s2v, stanceFeature], dim = 1) 

        # 采样生成保留边的编号，注意模型在eval模式下不应该dropedge
        if self.training:
            edgeNum = thread['edgeIndexTD'].shape[1]
            savedIndex = sample(range(edgeNum), int(edgeNum * (1 - self.edge_drop_rate)))
            savedIndex = sorted(savedIndex)
            edgeTD = [[thread['edgeIndexTD'][row][col] for col in savedIndex] for row in range(2)]
            edgeTD = torch.LongTensor(edgeTD)
        else:
            edgeTD = thread['edgeIndexTD']
        dataTD = Data(
            x = torch.clone(s2v).to(self.device), 
            edgeIndex = edgeTD.to(self.device), 
            rootIndex = thread['threadIndex']
        )
        if self.training:
            edgeNum = thread['edgeIndexTD'].shape[1]
            savedIndex = sample(range(edgeNum), int(edgeNum * (1 - self.edge_drop_rate)))
            savedIndex = sorted(savedIndex)
            edgeBU = [[thread['edgeIndexBU'][row][col] for col in savedIndex] for row in range(2)]
            edgeBU = torch.LongTensor(edgeBU)
        else:
            edgeBU = thread['edgeIndexBU']
        dataBU = Data(
            x = torch.clone(s2v).to(self.device), 
            edgeIndex = edgeBU.to(self.device), 
            rootIndex = thread['threadIndex']
        )

        rumorFeature = self.biGCN(dataTD, dataBU)

        return self.f2tRumor(rumorFeature), self.f2tStance(stanceFeature)

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