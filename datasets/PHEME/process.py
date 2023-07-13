import json
import os
import time
from copy import copy
from convert_veracity_annotations import convert_annotations
from tqdm import tqdm

DATA_PATH = '../../../rumorDataset/all-rnr-annotated-threads/'
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
    thread_id = []
    post_id = []
    thread_tag = {}
    structures = {}
    posts = {}
    eventName = os.listdir(DATA_PATH)
    for event in eventName:
        if '.' in event: continue

        ids = os.listdir(os.path.join(DATA_PATH, event, 'non-rumours'))
        for tid in tqdm(ids, desc='{} non-rumor'.format(event)):
            if '.' in tid: continue
            path = os.path.join(DATA_PATH, event, 'non-rumours', str(tid))
            
            # source tweet id
            thread_id.append(tid)
            post_id.append(tid)
            
            # thread tag(non-rumor)
            thread_tag[tid] = 0
            
            # structure
            with open(os.path.join(path, 'structure.json'), 'r') as f:
                content = f.read()
            structures[tid] = json.loads(content)
            # post
            with open(os.path.join(path, 'source-tweets', str(tid) + '.json'), 'r') as f:
                content = json.loads(f.read())
            post = {
                'time': strTime2Timestamp(content['created_at']),
                'text': content['text']
            }
            posts[tid] = post

            pids = os.listdir(os.path.join(path, 'reactions'))
            for pfname in pids:
                if '._' not in pfname and '.json' in pfname:
                    pid = pfname[0:-5]
                    post_id.append(pid)
                    with open(os.path.join(path, 'reactions', str(pfname)), 'r') as f:
                        content = json.loads(f.read())
                    post = {
                        'time': strTime2Timestamp(content['created_at']),
                        'text': content['text']
                    }
                    posts[pid] = post
            
                
        ids = os.listdir(os.path.join(DATA_PATH, event, 'rumours'))
        for tid in tqdm(ids, desc='{} rumor'.format(event)):
            if '.' in tid: continue
            path = os.path.join(DATA_PATH, event, 'rumours', str(tid))
            # id
            thread_id.append(tid)
            post_id.append(tid)
            # tag
            with open(os.path.join(path, 'annotation.json'), 'r') as f:
                annotation = json.loads(f.read())
                label = convert_annotations(annotation)
            thread_tag[tid] = label2Index[label]
            # structure
            with open(os.path.join(path, 'structure.json'), 'r') as f:
                content = f.read()
            structures[tid] = json.loads(content)
            # post
            with open(os.path.join(path, 'source-tweets', str(tid) + '.json'), 'r') as f:
                content = json.loads(f.read())
            post = {
                'time': strTime2Timestamp(content['created_at']),
                'text': content['text']
            }
            posts[tid] = post
            
            pids = os.listdir(os.path.join(path, 'reactions'))
            for pfname in pids:
                if '._' not in pfname and '.json' in pfname:
                    pid = pfname[0:-5]
                    post_id.append(pid)
                    with open(os.path.join(path, 'reactions', str(pfname)), 'r', encoding='utf8') as f:
                        content = json.loads(f.read())
                    post = {
                        'time': strTime2Timestamp(content['created_at']),
                        'text': content['text']
                    }
                    posts[pid] = post
            
    return thread_id, list(set(post_id)), thread_tag, structures, posts

# 转换字符串时间为时间戳
def strTime2Timestamp(strTime: str):
    temp = strTime.split(' ')
    temp.pop(-2) # 放弃掉不能读入的时区字段
    strTime = ' '.join(temp)
    structureTime = time.strptime(strTime, '%a %b %d %H:%M:%S %Y')
    return time.mktime(structureTime) # 把结构化时间转化成时间戳