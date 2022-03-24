'''
MT模型使用的semEval2017/PHEME数据集loader
'''
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import json
from gensim.models.keyedvectors import KeyedVectors
from utils import flattenStructure
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy

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
            time2Id = {}
            for id in ids:
                if id in rawDataset['posts']:
                    time2Id[str(rawDataset['posts'][id]['time'])] = id
            # post按照时间先后排序
            time2Id = sorted(time2Id.items(), key=lambda d: d[0])
            for (time, id) in time2Id:
                # 需要检查一下数据集中是否有这个id对应的post
                if id in rawDataset['posts'] and id in rawDataset['stanceTag']:
                    thread.append(rawDataset['posts'][id]['text'])
                    stanceTag.append(rawDataset['label2IndexStance'][rawDataset['stanceTag'][id]])
            
            self.dataset.append({
                'thread': thread,
                'rumorTag': torch.LongTensor(
                    [rawDataset['label2IndexRumor'][rawDataset['rumorTag'][threadId]]]
                ),
                'stanceTag': torch.LongTensor(stanceTag),
            })

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

# 把thread内的nodeFeature转化成Tensor，注意返回的需要是原数据的拷贝，不然会被更改
def collate(batch):
    thread = deepcopy(batch[0])

    return thread