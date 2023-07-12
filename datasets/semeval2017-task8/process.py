import json
import os
import time
from copy import copy
from tqdm import tqdm

DATA_PATH = '../../../rumorDataset/semeval2017/semeval2017-task8-dataset/'
TEST_SET_PATH = '../../../rumorDataset/semeval2017/semeval2017-task8-test-data/'
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

def readTrainPostIdAndTag():
    '''
    根据semeval2017 task8训练集的立场标签, 获取post的ID标识和立场标签
    Output:
        - post_index: list, post的ID
        - stance_label: dict{id: stance}, post的立场标注
    '''
    post_index = []
    file = open(os.path.join(DATA_PATH, 'traindev', 'rumoureval-subtaskA-train.json'), 'r')
    posts = json.load(file)
    for key in posts:
        post_index.append(key)
    file.close()
    #print(len(posts1))

    post_index.sort()
    stance_label = posts
    
    #print(len(stance_label))
    return post_index, stance_label

def readDevPostIdAndTag():
    '''
    根据semeval2017 task8验证集的立场标签, 获取post的ID标识和立场标签
    Output:
        - post_index: list, post的ID
        - stance_label: dict{id: stance}, post的立场标注
    '''
    post_index = []
    file = open(os.path.join(DATA_PATH, 'traindev', 'rumoureval-subtaskA-dev.json'), 'r')
    posts = json.load(file)
    for key in posts:
        post_index.append(key)
    file.close()
    #print(len(posts1))

    post_index.sort()
    stance_label = posts

    return post_index, stance_label

def readTrainthreadIndexAndTag():
    '''
    根据semeval2017 task8训练集的谣言标签获取thread的ID标识和谣言标签
    Output:
        - thread_index: list, post的ID
        - rumor_label: dict{id: stance}, post的立场标注
    '''
    thread_index = []
    file = open(os.path.join(DATA_PATH, 'traindev', 'rumoureval-subtaskB-train.json'), 'r')
    threads = json.load(file)
    for key in threads:
        thread_index.append(key)
    file.close()

    thread_index.sort()
    rumor_label = threads
    return thread_index, rumor_label

def readDevthreadIndexAndTag():
    '''
    根据semeval2017 task8训练集的谣言标签获取thread的ID标识和谣言标签
    Output:
        - thread_index: list, post的ID
        - rumor_label: dict{id: stance}, post的立场标注
    '''
    thread_index = []
    file = open(os.path.join(DATA_PATH, 'traindev', 'rumoureval-subtaskB-dev.json'), 'r')
    threads = json.load(file)
    for key in threads:
        thread_index.append(key)
    file.close()

    thread_index.sort()
    rumor_label = threads
    return thread_index, rumor_label

def readPostsStruct():
    '''
    读入所有post和传播树结构
    Output:
        - posts: dict{id: {time, text}}
        - structs: dict{id1: {id2: ..., id3:...}}
    '''
    posts = {}
    structs = {}
    for name in EVENT_NAME:
        folder_path = os.path.join(DATA_PATH, 'rumoureval-data', name)
        thread_index_list = os.listdir(folder_path)
        for thread_index in thread_index_list:
            struct = {}
            thread_path = os.path.join(folder_path, thread_index)
            # 尝试打开structure.json
            try:
                f = open(os.path.join(thread_path, 'structure.json'), 'r')
            except IOError:
                print('missing structure.json in path: {}'.format(thread_path))
                continue
            else:
                struct = json.load(f)
                structs[thread_index] = struct
                f.close()
            # 处理thread对应的文本和时间
            with open(os.path.join(thread_path, 'source-tweet', thread_index + '.json'), 'r') as file:
                data = json.load(file)
                post = {}
                post['time'] = strTime2Timestamp(data['created_at'])
                post['text'] = data['text']
                posts[thread_index] = post
            fileList = os.listdir(os.path.join(thread_path, 'replies'))
            for fileName in fileList:
                if '.json' in fileName:
                    with open(os.path.join(thread_path, 'replies', fileName), 'r') as file:
                        data = json.load(file)
                        post = {}
                        post['time'] = strTime2Timestamp(data['created_at'])
                        post['text'] = data['text']
                        posts[fileName[0:-5]] = post
    return posts, structs

def strTime2Timestamp(strTime: str):
    '''
    转换字符串时间为时间戳
    Input:
        - strTime: str, 时间格式举例: Tue Jan 07 11:38:00 +0000 2014
    Output:
        - float: 时间戳
    '''
    temp = strTime.split(' ')
    temp.pop(-2) # 放弃掉不能读入的时区字段
    strTime = ' '.join(temp)
    structureTime = time.strptime(strTime, '%a %b %d %H:%M:%S %Y')
    return time.mktime(structureTime) # 把结构化时间转化成时间戳
    
def readTestPostIdAndTag():
    '''
    根据semeval2017 task8测试集的立场标签, 获取post的ID标识和立场标签
    Output:
        - post_index: list, post的ID
        - stance_label: dict{id: stance}, post的立场标注
    '''
    post_index = []
    file = open(os.path.join(TEST_SET_PATH + 'subtaska.json'), 'r')
    stance_label = json.load(file)
    for key in stance_label:
        post_index.append(key) 
    file.close()

    post_index.sort()
    return post_index, stance_label

def readTestthreadIndexAndTag():
    '''
    根据semeval2017 task8测试集的谣言标签获取thread的ID标识和谣言标签
    Output:
        - thread_index: list, post的ID
        - rumor_label: dict{id: stance}, post的立场标注
    '''
    thread_index = []
    file = open(os.path.join(TEST_SET_PATH + 'subtaskb.json'), 'r')
    rumor_label = json.load(file)
    for key in rumor_label:
        thread_index.append(key)
    file.close()

    thread_index.sort()
    return thread_index, rumor_label

def readTestPostsStruct():
    '''
    读入测试集post和传播树结构
    Output:
        - posts: dict{id: {time, text}}
        - structs: dict{id1: {id2: ..., id3:...}}
    '''
    posts = {}
    structs = {}
    folder_path = TEST_SET_PATH
    thread_index_list = os.listdir(folder_path)
    for thread_index in thread_index_list:
        struct = {}
        thread_path = os.path.join(folder_path, thread_index)
        # 尝试打开structure.json
        try:
            f = open(os.path.join(thread_path, 'structure.json'), 'r')
        except IOError:
            print('missing structure.json in path: {}'.format(thread_path))
            continue
        else:
            struct = json.load(f)
            structs[thread_index] = struct
            f.close()
        # 处理thread对应的文本和时间
        with open(os.path.join(thread_path, 'source-tweet', thread_index + '.json'), 'r') as file:
            data = json.load(file)
            post = {}
            post['time'] = strTime2Timestamp(data['created_at'])
            post['text'] = data['text']
            posts[thread_index] = post
        fileList = os.listdir(os.path.join(thread_path, 'replies'))
        for fileName in fileList:
            if '.json' in fileName:
                with open(os.path.join(thread_path, 'replies', fileName), 'r') as file:
                    data = json.load(file)
                    post = {}
                    post['time'] = strTime2Timestamp(data['created_at'])
                    post['text'] = data['text']
                    posts[fileName[0:-5]] = post
    return posts, structs
            