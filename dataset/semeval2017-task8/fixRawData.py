import gensim
import json
from gensim.models.keyedvectors import KeyedVectors
from data import readPostIdAndTag, readThreadIdAndTag, readPostsStruct, readTestPostIdAndTag, readTestThreadIdAndTag, readTestPostsStruct
from utils import *

GLOVE_PATH = '../glove/' # glove预训练词嵌入的相对位置
WORD2VEC_DIMS = ['glove.twitter.27B.25d.gensim.txt', 
                 'glove.twitter.27B.50d.gensim.txt', 
                 'glove.twitter.27B.100d.gensim.txt', 
                 'glove.twitter.27B.200d.gensim.txt']

trainPostIds, trainStanceTag = readPostIdAndTag()
trainThreadIds, trainRumorTag = readThreadIdAndTag()
trainPosts, structures = readPostsStruct()
testPostIds, testStanceTag = readTestPostIdAndTag()
testThreadIds, trainRumorTag = readTestThreadIdAndTag()
testPosts, structures = readTestPostsStruct()
glove25d = KeyedVectors.load_word2vec_format(GLOVE_PATH + WORD2VEC_DIMS[0], binary=False)

trainSet = {'postIds': trainPostIds, 'stanceTag': trainStanceTag,
            'threadIds': trainThreadIds, 'rumorTag': trainRumorTag,
            'structures': structures}
testSet = {'postIds': testPostIds, 'stanceTag': testStanceTag,
           'threadIds': testThreadIds, 'rumorTag': trainRumorTag,
           'structures': structures}
for key in trainPosts:
    text = trainPosts[key]['text']
    words = fixText(text)
    for i in range(len(words)):
        if words[i] not in glove25d:
            words[i] = '<pad>'
    trainPosts[key]['text'] = ' '.join(words)
trainSet['posts'] = trainPosts
for key in testPosts:
    text = testPosts[key]['text']
    words = fixText(text)
    for i in range(len(words)):
        if words[i] not in glove25d:
            words[i] = '<pad>'
    testPosts[key]['text'] = ' '.join(words)
testSet['posts'] = testPosts
with open('trainSet.json', 'w') as f:
    f.write(json.dumps(trainSet))
with open('testSet.json', 'w') as f:
    f.write(json.dumps(testSet))