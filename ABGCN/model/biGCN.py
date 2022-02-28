import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, inputDim, hiddenDim, outDim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(inputDim, hiddenDim)
        self.conv2 = GCNConv(hiddenDim + inputDim, outDim)

    def forward(self, data):
        posts, edge_index, rootIndex = data.x, data.edge_index, data.rootIndex # posts(n, inputDim)
        
        conv1Out = self.conv1(posts, edge_index)
        postRoot = torch.copy(posts[rootIndex])
        postRoot = postRoot.repeat(posts.shape[0])
        conv1Root = conv1Out[rootIndex]

        conv2In = torch.cat([conv1Out, postRoot], dim=1)
        conv2In = F.relu(conv2In)
        conv2In = F.dropout(conv2In, training=self.training)
        conv2Out = self.conv2(conv2In, edge_index)
        conv2Out = F.relu(conv2Out)

        conv1Root = conv1Root.repeat(posts.shape[0])
        feature = torch.cat([conv1Root, conv2Out], dim=1)
        feature = feature.view(1, feature.shape[0], feature.shape[1])
        # avg_pool1d要求的输入是(batch, in_channels, *)
        feature = F.avg_pool1d(feature, kernel_size=3, stride=1) # 设置步长为1，使得卷积不会改变特征维度
        return feature.view(feature.shape[0], feature[1])
    
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
        return p[dataTD.rootIndex]

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