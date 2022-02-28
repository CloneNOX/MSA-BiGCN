import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import json
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
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
                for word in text:
                    if word in word2vec:
                        textW2v.append(word2vec[word].tolist())
                    else:
                        textW2v.append([0. for _ in range(w2vDim)])
                thread[i][1] = torch.Tensor(textW2v)
            self.dataset.append({
                'threadId': id,
                'threads': thread,
                'rumorTag': torch.LongTensor(
                    [rawDataset['label2IndexRumor'][rawDataset['rumorTag'][threadId]]]),
                'stanveTag': torch.LongTensor(stanceTag),
                'structure': structure
            })

    def __getitem__(self, item):
        return self.dataset[item]

    def _len__(self):
        return len(self.dataset)