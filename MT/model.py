import torch
import torch.nn as nn
from torch.nn.modules.module import Module

class MTUS(nn.Module):
    def __init__(self, inputDim: int, numRumorClass: int, numStanceClass: int,  
                 numGRULayer=2, embeddingDim=100, hiddenDim=100,
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
        self.shareGRU = nn.GRU(embeddingDim, hiddenDim, numGRULayer, bidirectional=self.bidirectional)
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
        gruOut, _ = self.shareGRU(embeddings, self.h0) # gruOut(seqLen, batch, numDirection * hiddenDim)
        h1 = gruOut[0].repeat(seqLen, 1, 1) # h1(seqLen, batch, numDirection * hiddenDim)
        p = self.vStance(torch.cat([h1, gruOut], dim=2))
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
    def __init__(self, inputDim: int, numRumorClass: int, numStanceClass: int,
                 numGRULayer=2, embeddingDim=100, hiddenDim=100, typeUS2M='cat',
                 batchSize = 1, bidirectional = False):
        super().__init__() # 调用nn.Moudle父类的初始化方法
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 优先使用cuda
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional else 1
        self.typeUS2M = typeUS2M
        
        # embedding使用线性层来把tf-idf向量转换成句嵌入
        self.embeddingRumor = nn.Linear(inputDim, embeddingDim)
        self.embeddingStance = nn.Linear(inputDim, embeddingDim)

        # 共享GRU层
        self.GRUshare = nn.GRU(embeddingDim, hiddenDim, numGRULayer, bidirectional=self.bidirectional)
        self.h0Share = nn.Parameter(torch.randn((self.D * numGRULayer, batchSize, hiddenDim)))

        # 非共享GRU

        if self.typeUS2M == 'add':
        # ======相加======
            self.GRURumor = nn.GRU(hiddenDim, hiddenDim, numGRULayer, bidirectional=self.bidirectional)
            self.GRUStance = nn.GRU(hiddenDim, hiddenDim, numGRULayer, bidirectional=self.bidirectional)
        # ================

        else:
        # ======拼接======
            self.GRURumor = nn.GRU(2 * hiddenDim, hiddenDim, numGRULayer, bidirectional=self.bidirectional)
            self.GRUStance = nn.GRU(2 * hiddenDim, hiddenDim, numGRULayer, bidirectional=self.bidirectional)
        # ================

        self.h0Rumor = nn.Parameter(torch.randn((self.D * numGRULayer, batchSize, hiddenDim)))
        self.h0Stance = nn.Parameter(torch.randn((self.D * numGRULayer, batchSize, hiddenDim)))

        # 共享GRU隐状态接入非共享层的系数

        if self.typeUS2M == 'add':
        # ======相加======
            self.US2M = nn.Linear(self.D * hiddenDim, embeddingDim)
        # ================

        else:
        # ======拼接======
            self.US2M = nn.Linear(2 * self.D * hiddenDim, embeddingDim)
        # ================


        # 把GRU的隐状态映射成概率
        self.vRumor = nn.Linear(self.D * hiddenDim, numRumorClass)
        self.vStance = nn.Linear(self.D * 2 * hiddenDim, numStanceClass) # stance的预测需要拼接第一句的隐状态
        
        # 训练集前向传递，返回对特定任务的概率向量/矩阵
    def forwardRumor(self, sentences: torch.Tensor):
        seqLen = sentences.size()[0]
        embeddings = self.embeddingRumor(sentences).view(seqLen, self.batchSize, self.embeddingDim) # embeddings(sepLen, batch, embeddingDim)   
        gruShareOut, _ = self.GRUshare(embeddings, self.h0Share)
        # gruShareOut(seqLen, batch, numDirection * hiddenDim)

        if self.typeUS2M == 'add':
        # ======拼接======
            hS2M = self.US2M(gruShareOut) # hS2M(sepLen, batch, embeddingDim)
            gruRumorIn = embeddings + hS2M
        # ================

        else:
        # ======相加======
            gruRumorIn = torch.cat([embeddings, gruShareOut], dim=2)
        # ================

        gruRumorOut, _ = self.GRURumor(gruRumorIn, self.h0Rumor)
        # gruRumorOut(seqLen, batch, numDirection * hiddenDim)

        ht = gruRumorOut[gruRumorOut.size()[0] - 1].view(self.batchSize, self.D * self.hiddenDim) # 取出最后一层的隐状态
        p = self.vRumor(ht)
        return p
        
    def forwardStance(self, sentences: torch.Tensor):
        seqLen = sentences.size()[0]
        embeddings = self.embeddingRumor(sentences).view(seqLen, self.batchSize, self.embeddingDim)
        gruShareOut, _ = self.GRUshare(embeddings, self.h0Share)
        # gruShareOut(seqLen, batch, numDirection * hiddenDim)
        
        if self.typeUS2M == 'add':
        # ======相加======
            hS2M = self.US2M(gruShareOut) # hS2M(sepLen, batch, embeddingDim)
            gruStanceIn = embeddings + hS2M
        # ================

        else:
        # ======拼接======
            gruStanceIn = torch.cat([embeddings, gruShareOut], dim=2)
        # ================

        gruStanceOut, _ = self.GRUStance(gruStanceIn, self.h0Stance)
        # gruRumorOut(seqLen, batch, numDirection * hiddenDim)

        h1 = gruStanceOut[0].repeat(seqLen, 1, 1) # h1(seqLen, batch, numDirection * hiddenDim)
        ps = self.vStance(torch.cat([h1, gruStanceOut], dim=2))
        return ps
        
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
