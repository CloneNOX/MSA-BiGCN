import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from biGCN import *

class ABGCN(nn.Module):
    def __init__(self,
                 w2vDim: int,
                 s2vDim: int,
                 gcnHiddenDim: int,
                 rumorFeatureDim: int,
                 batchSize = 1,
                 s2vMethon = 'a',
                 numLstmLayer = 2):
        super().__init__()
        self.w2vDim = w2vDim
        self.s2vDim = s2vDim
        self.gcnHiddenDim = gcnHiddenDim
        self.rumorFeatureDim = rumorFeatureDim
        self.s2vMethon = s2vMethon
        self.batchSize = batchSize
        
        # ==使用biLSTM获取sentence to vector
        if self.s2vMethon == 'l':
            self.numLstmLayer = numLstmLayer
            self.lstm = nn.LSTM(input_size = w2vDim,
                                hidden_size = s2vDim,
                                num_layers = numLstmLayer,
                                bidirectional=True)
            self.h0 = nn.Parameter(torch.randn((2 * 2, batchSize, s2vDim)))
            self.c0 = nn.Parameter(torch.randn((2 * 2, batchSize, s2vDim)))
        
        self.biGCN = BiGCN()

    def forward(self, dataTD, dataBU):
        pass