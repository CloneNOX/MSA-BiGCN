import gensim
from gensim.models.keyedvectors import KeyedVectors

# 修改输入的字符串：字母转换为小写，符号/数字与字母分开。返回List[str]
def fixText(raw: str):
    character = 'abcdefghijklmnopqrstuvwxyz'
    words = raw.split(' ')
    fixed = []
    for word in words:
        word = word.lower()
        start = 0
        i = 0
        while i in range(len(word)):
            if word[i] not in character:
                if(len(word[start : i]) != 0):
                    fixed.append(word[start : i])
                start = i
                while i in range(len(word)) and word[i] not in character:
                    i += 1
                if(len(word[start : i]) != 0):
                    fixed.append(word[start : i])
                start = i
            i += 1
        if(len(word[start : i]) != 0):
            fixed.append(word[start : i + 1])
    return fixed
