import torch
import torch.nn as nn
from torch.nn.modules.module import Module

class MTUS(nn.Module):
    def __init__(self, embeddingDim: int, hiddenDim: int, inputDim: int, 
                 numGRULayer: int, numRumorClass: int, numStanceClass: int,
                 batchSize = 1, bidirectional = False):
        super().__init__() # 调用nn.Moudle父类的初始化方法
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 优先使用cuda
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional else 1
        
        # embedding使用线性层来把tf-idf向量转换成句嵌入
        self.embeddingRumor = nn.Linear(inputDim, embeddingDim)
        self.embeddingStance = nn.Linear(inputDim, embeddingDim)
        
        # 共享GRU层
        self.shareGRU = nn.GRU(embeddingDim, hiddenDim, numGRULayer, bidirectional = self.bidirectional)
        self.h0 = nn.Parameter(torch.randn((self.D * numGRULayer, self.batchSize, hiddenDim)))

        # 把GRU的隐状态映射成概率
        self.vRumor = nn.Linear(self.D * hiddenDim, numRumorClass)
        self.vStance = nn.Linear(self.D * 2 * hiddenDim, numStanceClass) # stance的预测需要拼接第一句的隐状态
        
    # 训练集前向传递，返回对特定任务的概率向量/矩阵
    def forwardRumor(self, sentences: torch.Tensor):
        seqLen = sentences.size()[0] # 取tensor size的第一维，是本次训练的thread的长度
        embeddings = self.embeddingRumor(sentences).view(seqLen, self.batchSize, self.embeddingDim) # view是为了适配gru的输入样式
        # hs(seqLen, batch, numDirection * hiddenDim), ht(numLayers*numDirections, batch, hiddenDim)
        # 舍弃掉ht的输出是因为下一个thread和这次训练的thread是独立的，不应该用本次的隐状态作为其h0输入
        gruOut, _ = self.shareGRU(embeddings, self.h0) 
        ht = gruOut[gruOut.size()[0] - 1].view(self.batchSize, self.D * self.hiddenDim) # 取出最后一层的隐状态
        p = self.vRumor(ht)
        return p # 返回的概率矩阵是包含batch维度的size():(batch, numDirection)
    
    def forwardStance(self, sentences: torch.Tensor):
        seqLen = sentences.size()[0]
        embeddings = self.embeddingRumor(sentences).view(seqLen, self.batchSize, self.embeddingDim)
        gruOut, _ = self.shareGRU(embeddings, self.h0) # hs(seqLen, batch, numDirection * hiddenDim)
        h1Repeat = gruOut[0].repeat(seqLen, 1, 1) # h1Repeat(seqLen, batch, numDirection * hiddenDim)
        p = self.vStance(torch.cat([h1Repeat, gruOut], dim=2))
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
    
class MTES(nn.Module):
    def __init__(self, embeddingDim: int, hiddenDim: int, inputDim: int, numGRULayer: int, numRumorClass: int, numStanceClass: int, batchSize=1):
        super().__init__() # 调用nn.Moudle父类的初始化方法
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 优先使用cuda
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize
        
        # embedding使用线性层来把tf-idf向量转换成句嵌入
        self.embeddingRumor = nn.Linear(inputDim, embeddingDim)
        self.embeddingStance = nn.Linear(inputDim, embeddingDim)
        # 共享GRU层
        self.shareGRU = nn.GRU(embeddingDim, hiddenDim, numGRULayer)
        self.h0 = nn.Parameter(torch.randn((numGRULayer, batchSize, hiddenDim)))
        # 非共享GRU
        self.GRURumor = nn.GRU(hiddenDim, hiddenDim, numGRULayer)
        self.GRUStance = nn.GRU(hiddenDim, hiddenDim, numGRULayer)
        self.h0Rumor = nn.Parameter(torch.randn((numGRULayer, batchSize, hiddenDim)))
        self.h0Stance = nn.Parameter(torch.randn((numGRULayer, batchSize, hiddenDim)))
        # 共享GRU隐状态接入非共享层的系数
        self.u = nn.Linear(hiddenDim, embeddingDim)
        # 把GRU的隐状态映射成概率
        self.vRumor = nn.Linear(hiddenDim, numRumorClass)
        self.v1Stance = nn.Linear(hiddenDim, numStanceClass)
        self.vStance = nn.Linear(hiddenDim, numStanceClass)
        
        # 训练集前向传递，返回对特定任务的概率向量/矩阵
    def forwardRumor(self, sentences: torch.Tensor):
        seqLen = sentences.size()[0]
        sentence_len = sentences.size()[1]
        embeddings = self.embeddingRumor(sentences).view(seqLen, self.batchSize, self.embeddingDim)
        hs, _ = self.shareGRU(embeddings, self.h0)
        gru_Rumor_in = embeddings + self.u(hs)
        hs_Rumor, ht = self.GRURumor(gru_Rumor_in, self.h0Rumor)
        ht = ht[ht.size()[0] - 1].view(self.hiddenDim)
        p = self.vRumor(ht)
        return p
        
    def forwardStance(self, sentences: torch.Tensor):
        seqLen = sentences.size()[0]
        sentence_len = sentences.size()[1]
        embeddings = self.embeddingRumor(sentences).view(seqLen, self.batchSize, self.embeddingDim)
        hs, _ = self.shareGRU(embeddings, self.h0)
        gru_Stance_in = embeddings + self.u(hs)
        hs_Stance, ht = self.GRURumor(gru_Stance_in, self.h0Rumor)
        hs_Stance = hs_Stance.view(seqLen, self.hiddenDim)
        ps = self.v1Stance(hs_Stance[0]) + self.vStance(hs_Stance)
        return ps
        
    # 更换计算设备
    def set_device(self, device: torch.device) -> torch.nn.Module:
        _model = self.to(device)
        _model.device = device
        return _model
