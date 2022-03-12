import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import json
from gensim.models.keyedvectors import KeyedVectors
from utils import flattenStructure
from torch.nn.utils.rnn import pad_sequence
from copy import copy

class semEval2017Dataset(Dataset):
    def __init__(self, dataPath: str, type: str) -> None:
        with open(dataPath + type + 'Set.json', 'r') as f:
            content = f.read()
        rawDataset = json.loads(content)

        self.dataset = []

        for threadId in rawDataset['threadIds']:
            thread = []
            stanceTag = []
            structure = rawDataset['structures'][threadId]
            ids = flattenStructure(structure)
            for id in ids:
                # 需要检查一下数据集中是否有这个id对应的post
                if id in rawDataset['posts'] and id in rawDataset['stanceTag']:
                    thread.append([id, rawDataset['posts'][id]['text']])
                    stanceTag.append(rawDataset['label2IndexStance'][rawDataset['stanceTag'][id]])
            threadIndex, nodeText, edgeIndex = self.structure2graph(threadId, thread, structure)
            self.dataset.append({
                'threadId': id,
                'threadIndex': threadIndex,
                'nodeText': nodeText,
                'edgeIndexTD': edgeIndex,
                'edgeIndexBU': torch.flip(edgeIndex, dims=[0]),
                'rumorTag': torch.LongTensor(
                    [rawDataset['label2IndexRumor'][rawDataset['rumorTag'][threadId]]]),
                'stanceTag': torch.LongTensor(stanceTag),
            })

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    # 根据输入的结构dict生成图，返回：threadIndex：thread在nodeFeature中的下标，nodeFeature，edgeIndex(TD)
    def structure2graph(self, threadId, thread, structure):
        threadIndex = 0
        nodeText = []
        id2Index = {}
        edgeIndex = [[], []]

        for i in range(len(thread)): # w2v(seqLen, w2vDim)这里的seqLen是不一致的
            if thread[i][0] == threadId:
                threadIndex = i
            nodeText.append(thread[i][1])
            id2Index[thread[i][0]] = i

        # 使用栈模拟递归建立TD树状图
        stack = [[None, structure]]
        while stack:
            parent, childDict = stack[-1]
            if parent != None:
                for child in childDict:
                    if parent in id2Index and child in id2Index:
                        edgeIndex[0].append(id2Index[parent])
                        edgeIndex[1].append(id2Index[child])
            
            stack.pop()
            for child in childDict:
                if childDict[child]:
                    stack.append([child, childDict[child]])
        edgeIndex = torch.LongTensor(edgeIndex)
        
        return threadIndex, nodeText, edgeIndex

# 把thread内的nodeFeature转化成Tensor，注意返回的需要是原数据的拷贝，不然会被更改
def collate(batch, word2Vec, cls: torch.Tensor):
    thread = batch[0]
    nodeText = []
    for text in thread['nodeText']: # nf: list[word, word...]
        nodeText.append("<start>" + text + "<end>")
    thread['nodeText'] = nodeText
    return thread