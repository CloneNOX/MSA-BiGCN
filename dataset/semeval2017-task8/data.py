import torch
import json
import os
import time
from typing import Dict, List

TRAINTAG_PATH = "./semeval2017-task8-dataset/traindev/"
DEVTAG_PATH = "./semeval2017-task8-dataset/traindev/"
TESTTAG_PATH = "./semeval2017-task8-test-data/"
TRAINDEV_DATA_PATH = './semeval2017-task8-dataset/rumoureval-data/'
TEST_DATA_PATH = "./semeval2017-task8-test-data/"

EVENT_NAME = ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege']

label2IndexRumor = {'unverified': 0, 'true': 1, 'false': 2}
label2IndexStance = {'support': 0, 'query': 1, 'deny': 2, 'comment': 3}
index2LabelRumor = {'0': 'unverified', '1': 'true', '2': 'false'}
index2LabelStance = {'0': 'support', '1': 'query', '2': 'deny', '3': 'comment'}

# 根据semeval2017 task8数据集的立场标签，获取post的ID标识和立场标签
def readTrainPostIdAndTag():
    postIds = []
    file = open(TRAINTAG_PATH + 'rumoureval-subtaskA-train.json', 'r')
    posts = json.load(file)
    for key in posts:
        postIds.append(key)
    file.close()
    #print(len(posts1))

    postIds.sort()
    stanceTag = posts
    
    #print(len(stanceTag))
    return postIds, stanceTag

def readDevPostIdAndTag():
    postIds = []
    file = open(TRAINTAG_PATH + 'rumoureval-subtaskA-dev.json', 'r')
    posts = json.load(file)
    for key in posts:
        postIds.append(key)
    file.close()
    #print(len(posts1))

    postIds.sort()
    stanceTag = posts

    return postIds, stanceTag

# 根据semeval2017 task8数据集的谣言标签，获取thread的ID标识和谣言标签
def readTrainThreadIdAndTag() -> List[str]:
    threadsId = []
    file = open(TRAINTAG_PATH + 'rumoureval-subtaskB-train.json', 'r')
    threads = json.load(file)
    for key in threads:
        threadsId.append(key)
    file.close()

    threadsId.sort()
    rumorTag = threads
    return threadsId, rumorTag

def readDevThreadIdAndTag() -> List[str]:
    threadsId = []
    file = open(TRAINTAG_PATH + 'rumoureval-subtaskB-dev.json', 'r')
    threads = json.load(file)
    for key in threads:
        threadsId.append(key)
    file.close()

    threadsId.sort()
    rumorTag = threads
    return threadsId, rumorTag

# 读入所有post，生成dict{id: {time, text}}
def readPostsStruct():
    posts = {}
    structs = {}
    for name in EVENT_NAME:
        folderPath = TRAINDEV_DATA_PATH + name
        threadIdList = os.listdir(folderPath)
        for threadId in threadIdList:
            struct = {}
            threadPath = folderPath + '/' + threadId + '/'
            # 尝试打开structure.json
            try:
                f = open(threadPath + 'structure.json', 'r')
            except IOError:
                continue
            else:
                struct = json.load(f)
                structs[threadId] = struct
                f.close()
            # 处理thread对应的文本和时间
            with open(threadPath + 'source-tweet/' + threadId + '.json', 'r') as file:
                data = json.load(file)
                post = {}
                post['time'] = strTime2Timestamp(data['created_at'])
                post['text'] = data['text']
                posts[threadId] = post
            fileList = os.listdir(threadPath + '/replies')
            for fileName in fileList:
                if '.json' in fileName:
                    with open(threadPath + '/replies/' + fileName, 'r') as file:
                        data = json.load(file)
                        post = {}
                        post['time'] = strTime2Timestamp(data['created_at'])
                        post['text'] = data['text']
                        posts[fileName[0:-5]] = post
    return posts, structs

# 转换字符串时间为时间戳
def strTime2Timestamp(strTime: str):
    temp = strTime.split(' ')
    temp.pop(-2) # 放弃掉不能读入的时区字段
    strTime = ' '.join(temp)
    structureTime = time.strptime(strTime, '%a %b %d %H:%M:%S %Y')
    return time.mktime(structureTime) # 把结构化时间转化成时间戳
    
# 根据semeval2017 task8测试集的立场标签，获取post的ID标识和立场标签
def readTestPostIdAndTag():
    postIds = []
    file = open(TESTTAG_PATH + 'testStance.json', 'r')
    stanceTag = json.load(file)
    for key in stanceTag:
        postIds.append(key)
    file.close()

    postIds.sort()
    return postIds, stanceTag

# 根据semeval2017 task8测试集的谣言标签，获取thread的ID标识和谣言标签
def readTestThreadIdAndTag() -> List[str]:
    threadsId = []
    file = open(TESTTAG_PATH + 'testRumor.json', 'r')
    rumorTag = json.load(file)
    for key in rumorTag:
        threadsId.append(key)
    file.close()

    threadsId.sort()
    return threadsId, rumorTag

# 读入测试集所有post，生成dict{id: {time, text}}
def readTestPostsStruct():
    posts = {}
    structs = {}
    folderPath = TEST_DATA_PATH
    threadIdList = os.listdir(folderPath)
    for threadId in threadIdList:
        struct = {}
        threadPath = folderPath + '/' + threadId + '/'
        # 尝试打开structure.json
        try:
            f = open(threadPath + 'structure.json', 'r')
        except IOError:
            continue
        else:
            struct = json.load(f)
            structs[threadId] = struct
            f.close()
        # 处理thread对应的文本和时间
        with open(threadPath + 'source-tweet/' + threadId + '.json', 'r') as file:
            data = json.load(file)
            post = {}
            post['time'] = strTime2Timestamp(data['created_at'])
            post['text'] = data['text']
            posts[threadId] = post
        fileList = os.listdir(threadPath + '/replies')
        for fileName in fileList:
            if '.json' in fileName:
                with open(threadPath + '/replies/' + fileName, 'r') as file:
                    data = json.load(file)
                    post = {}
                    post['time'] = strTime2Timestamp(data['created_at'])
                    post['text'] = data['text']
                    posts[fileName[0:-5]] = post
    return posts, structs
            