import json
import os
import time
from copy import copy
from convert_veracity_annotations import convert_annotations

DATA_PATH = './all-rnr-annotated-threads/'
EVENT_NAME = [
    'charliehebdo', 
    'ebola-essien', 
    'ferguson', 
    'germanwings-crash', 
    'ottawashooting', 
    'prince-toronto', 
    'putinmissing', 
    'sydneysiege'
]

label2Index = {'non-rumor': 0, 'true': 1, 'false': 2, 'unverified': 3}
index2Label = {'0': 'non-rumor', '1': 'true', '2': 'false', '3': 'unverified'}

def ProcessRNRDataset():
    threadId = []
    postId = []
    threadTag = {}
    structures = {}
    posts = {}
    eventName = os.listdir(DATA_PATH)
    for event in eventName:
        if '.' in event: continue

        ids = os.listdir(os.path.join(DATA_PATH, event, 'non-rumours'))
        for id in ids:
            if '.' in id: continue
            path = os.path.join(DATA_PATH, event, 'non-rumours', str(id))
            
            # source tweet id
            threadId.append(id)
            postId.append(id)
            
            # thread tag(non-rumor)
            threadTag[id] = 0
            
            # structure
            with open(os.path.join(path, 'structure.json'), 'r') as f:
                content = f.read()
            structures[id] = json.loads(content)
            # post
            post = {}
            with open(os.path.join(path, 'source-tweets', str(id) + '.json'), 'r') as f:
                content = json.loads(f.read())
            post[id] = {
                'time': strTime2Timestamp(content['created_at']),
                'text': content['text']
            }
            pids = os.listdir(os.path.join(path, 'reactions'))
            for pfname in pids:
                if '._' not in pfname and '.json' in pfname:
                    pid = pfname[0:-5]
                    postId.append(pid)
                    with open(os.path.join(path, 'reactions', str(pfname)), 'r') as f:
                        content = json.loads(f.read())
                    post[pid] = {
                        'time': strTime2Timestamp(content['created_at']),
                        'text': content['text']
                    }
            posts[id] = post
            break
                
        ids = os.listdir(os.path.join(DATA_PATH, event, 'rumours'))
        for id in ids:
            if '.' in id: continue
            path = os.path.join(DATA_PATH, event, 'rumours', str(id))
            # id
            threadId.append(id)
            postId.append(id)
            # tag
            with open(os.path.join(path, 'annotation.json'), 'r') as f:
                annotation = json.loads(f.read())
                label = convert_annotations(annotation)
            threadTag[id] = label2Index[label]
            # structure
            with open(os.path.join(path, 'structure.json'), 'r') as f:
                content = f.read()
            structures[id] = json.loads(content)
            # post
            post = {}
            with open(os.path.join(path, 'source-tweets', str(id), '.json'), 'r') as f:
                content = json.loads(f.read())
                post[id] = {
                    'time': strTime2Timestamp(content['created_at']),
                    'text': content['text']
                }
            pids = os.listdir(os.path.join(path, 'reactions'))
            for pfname in pids:
                if '._' not in pfname and '.json' in pfname:
                    pid = pfname[0:-5]
                    postId.append(pid)
                    with open(os.path.join(path, 'reactions', str(pfname)), 'r', encoding='utf8') as f:
                        content = json.loads(f.read())
                    post[pid] = {
                        'time': strTime2Timestamp(content['created_at']),
                        'text': content['text']
                    }
            posts[id] = post
    return threadId, postId, threadTag, structures, posts

# 转换字符串时间为时间戳
def strTime2Timestamp(strTime: str):
    temp = strTime.split(' ')
    temp.pop(-2) # 放弃掉不能读入的时区字段
    strTime = ' '.join(temp)
    structureTime = time.strptime(strTime, '%a %b %d %H:%M:%S %Y')
    return time.mktime(structureTime) # 把结构化时间转化成时间戳