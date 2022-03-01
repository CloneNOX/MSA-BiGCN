import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import json
from gensim.models.keyedvectors import KeyedVectors
from utils import flattenStructure

class semeval2017Dataset(Dataset):
    def __init__(self, dataPath: str, type: str, w2vPath=str, w2vDim=100) -> None:
        with open(dataPath + type + 'Set.json', 'r') as f:
            content = f.read()
        rawDataset = json.loads(content)

        word2vec = KeyedVectors.load_word2vec_format(
            w2vPath + 'glove.twitter.27B.' + str(w2vDim) + 'd.gensim.txt', binary=False)

        self.dataset = []

        for threadId in rawDataset['threadIds']:
            thread = []
            stanceTag = []
            structure = rawDataset['structures'][threadId]
            ids = flattenStructure(structure)
            for id in ids:
                if id in rawDataset['posts'] and id in rawDataset['stanceTag']:
                    thread.append([id, rawDataset['posts'][id]['text']])
                    stanceTag.append(rawDataset['label2IndexStance'][rawDataset['stanceTag'][id]])
            for i in range(len(thread)):
                text = thread[i][1]
                textW2v = []
                for word in text.split(' '):
                    if word in word2vec:
                        textW2v.append(word2vec[word].tolist())
                    else:
                        textW2v.append([0. for _ in range(w2vDim)])
                thread[i][1] = torch.Tensor(textW2v)
            threadIndex, nodeFeature, edgeIndex = self.structure2graph(threadId, thread, structure)
            self.dataset.append({
                'threadId': id,
                'threadIndex': threadIndex,
                'nodeFeature': nodeFeature,
                'edgeIndexTD': edgeIndex,
                'edgeIndexBU': torch.flip(edgeIndex, dims=[0]),
                'rumorTag': torch.LongTensor(
                    [rawDataset['label2IndexRumor'][rawDataset['rumorTag'][threadId]]]),
                'stanveTag': torch.LongTensor(stanceTag),
                'label2IndexRumor': rawDataset['label2IndexRumor'],
                'label2IndexStance': rawDataset['label2IndexStance']
            })

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    # 根据输入的结构dict生成图，返回：threadIndex：thread在nodeFeature中的下标，nodeFeature，edgeIndex(TD)
    def structure2graph(self, threadId, thread, structure):
        threadIndex = 0
        nodeFeature = []
        id2Index = {}
        edgeIndex = [[], []]

        for i in range(len(thread)): # w2v(seqLen, w2vDim)这里的seqLen是不一致的
            if thread[i][0] == threadId:
                threadIndex = i
            nodeFeature.append(thread[i][1])
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
        
        return threadIndex, nodeFeature, edgeIndex