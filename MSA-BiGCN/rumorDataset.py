import torch
from torch.utils.data import Dataset
import json
from utils import flattenStructure
import os
from transformers import BertTokenizer

THREAD_ID_FILE = 'thread_id.txt'
POST_ID_FILE = 'post_id.txt'
THREAD_LABEL_FILE = 'thread_label.txt'
STRUCTURE_FILE = 'structures.json'
POST_FILE = 'posts.json'
CATEGPRY = 'category.json'

class RumorDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer) -> None:
        '''
        PHEME数据集
        Input:
            - dataPath: 数据集预处理后文件目录
            - tokenizer: 词划分器
        '''
        with open(os.path.join(data_path, THREAD_ID_FILE)) as f:
            self.all_thread_id = [line.strip() for line in f.readlines()] # list
        with open(os.path.join(data_path, POST_ID_FILE)) as f:
            self.all_post_id = [line.strip() for line in f.readlines()] # list
        with open(os.path.join(data_path, THREAD_LABEL_FILE)) as f:
            self.all_thread_label = [line.strip() for line in f.readlines()] # list, 读入的是str形式
        with open(os.path.join(data_path, STRUCTURE_FILE)) as f:
            self.all_structure = json.load(f) # dict
        with open(os.path.join(data_path, POST_FILE)) as f:
            self.all_post = json.load(f) # dict
        with open(os.path.join(data_path, CATEGPRY)) as f:
            self.category = json.load(f) #dict

        self.tokenizer = tokenizer

    def __getitem__(self, item):
        thread_id = self.all_thread_id[item]
        structure = self.all_structure[thread_id]
        label = self.all_thread_label[item]
        
        # 获取所有的post id
        ids = flattenStructure(structure)
        
        thread = []
        for id in ids:
            # 需要检查一下数据集中是否有这个id对应的post
            if id in self.all_post_id:
                thread.append([id, self.all_post[thread_id][id]['text']])
        root_index, node_text, edge_index_TD = self.structure2graph(thread_id, thread, structure)
        return thread_id, root_index, node_text, edge_index_TD, label, self.tokenizer

    def __len__(self):
        return len(self.all_thread_id)

    def structure2graph(self, threadId: str, thread: list, structure: dict):
        '''
        根据输入的结构dict生成图, 
        Input:
            - threadId: 传播树的id
            - thread: [[id, post]]
            - structure: 传播树结构
        Output: 
            - root_index: thread(传播树根节点)在nodeFeature中的下标
            - node_text: 各个节点的文本
            - edge_index(TD): 边的
        '''
        root_index = 0 
        node_text = []
        id2Index = {}
        edge_index = [[], []]

        for i in range(len(thread)): # w2v(seqLen, w2vDim)这里的seqLen是不一致的
            if thread[i][0] == threadId:
                root_index = i
            node_text.append(thread[i][1])
            id2Index[thread[i][0]] = i

        # 使用栈模拟递归建立TD树状图
        stack = [[None, structure]]
        while stack:
            parent, childDict = stack[-1]
            if parent != None:
                for child in childDict:
                    if parent in id2Index and child in id2Index:
                        edge_index[0].append(id2Index[parent])
                        edge_index[1].append(id2Index[child])
            
            stack.pop()
            for child in childDict:
                if childDict[child]:
                    stack.append([child, childDict[child]])
        
        return root_index, node_text, edge_index
    
    def collate_fn(batch):
        '''
        "merges a list of samples to form a mini-batch of Tensor(s)"--pytorch
        Input:
            - batch: [(thread_id: str, 
                      root_index: int, 
                      node_text: [str], 
                      edge_index_TD: [[int], [int]], 
                      label: str),
                      tokenizer: BertTokenizer]
        Output: 将一个thread视作一个batch
            - node_token: transformers.BertTokenizer类的输出, 一般是, 未转成句向量
            - edge_index_TD: [[int]. [int]], [2, |E|], 自顶向下的传播树图
            - edge_index_BU: [[int]. [int]], [2, |E|], 自底向上的传播树图
            - root_index: int
            - label: Tensor, shape = [1]
        '''
        for item in batch:
            root_index = item[1]
            node_text = item[2]
            edge_index_TD = item[3]
            label = item[4]
            tokenizer = item[5]

        node_token = tokenizer(node_text, return_tensors='pt', padding=True)
        edge_index_BU = [edge_index_TD[1], edge_index_TD[0]]

        return node_token, edge_index_TD, edge_index_BU, root_index, torch.LongTensor([int(label)])

class RumorStanceDataset(Dataset):
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
            root_index, node_text, edge_index = self.structure2graph(threadId, thread, structure)
            self.dataset.append({
                'threadId': id,
                'root_index': root_index,
                'node_text': node_text,
                'edge_index_TD': edge_index,
                'edgeIndexBU': torch.flip(edge_index, dims=[0]),
                'rumorTag': torch.LongTensor(
                    [rawDataset['label2IndexRumor'][rawDataset['rumorTag'][threadId]]]),
                'stanceTag': torch.LongTensor(stanceTag),
            })

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    # 根据输入的结构dict生成图，返回：root_index：thread在nodeFeature中的下标，nodeFeature，edgeIndex(TD)
    def structure2graph(self, threadId, thread, structure):
        root_index = 0
        node_text = []
        id2Index = {}
        edge_index = [[], []]

        for i in range(len(thread)): # w2v(seqLen, w2vDim)这里的seqLen是不一致的
            if thread[i][0] == threadId:
                root_index = i
            node_text.append(thread[i][1])
            id2Index[thread[i][0]] = i

        # 使用栈模拟递归建立TD树状图
        stack = [[None, structure]]
        while stack:
            parent, childDict = stack[-1]
            if parent != None:
                for child in childDict:
                    if parent in id2Index and child in id2Index:
                        edge_index[0].append(id2Index[parent])
                        edge_index[1].append(id2Index[child])
            
            stack.pop()
            for child in childDict:
                if childDict[child]:
                    stack.append([child, childDict[child]])
        edge_index = torch.LongTensor(edge_index)
        
        return root_index, node_text, edge_index
