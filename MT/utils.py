import json
from sklearn.feature_extraction.text import TfidfVectorizer

def flattenStructure(structure: dict):
    Ids = []
    if not structure:
        return Ids # 当前dict为空，直接返回空列表
    Ids += list(structure.keys())
    for id in structure:
        if structure[id]: # 子dict不为空，递归地展开
            Ids += flattenStructure(structure[id])
    return Ids
    
def getTfidfAndLabel(dataSetPath:str, dim=5000):
    with open(dataSetPath + 'trainSet.json', 'r') as f:
        content = f.read()
    trainSet = json.loads(content)
    threads = []
    for threadId in trainSet['threadIds']:
        thread = []
        structure = trainSet['structures'][threadId]
        ids = flattenStructure(structure)
        time2Id = {}
        for id in ids:
            if id in trainSet['posts']:
                time2Id[str(trainSet['posts'][id]['time'])] = id
        # post按照时间先后排序
        time2Id = sorted(time2Id.items(), key=lambda d: d[0])
        for (time, id) in time2Id:
            if id in trainSet['posts']:
                thread.append(trainSet['posts'][id]['text'])
        threads.append(thread)
    cropus = []
    for thread in threads:
        for text in thread:
            cropus.append(text)
    tfidf_vec = TfidfVectorizer(max_features = dim)
    tfidf_vec.fit(cropus)


    if 'PHEME' in dataSetPath:
        label2IndexRumor = trainSet['label2Index'] if 'label2Index' in trainSet else None
        label2IndexStance = None
    else:
        label2IndexRumor = trainSet['label2IndexRumor'] if 'label2IndexRumor' in trainSet else None
        label2IndexStance = trainSet['label2IndexStance'] if 'label2IndexStance' in trainSet else None

    return tfidf_vec, label2IndexRumor, label2IndexStance
# end getTrainSet()