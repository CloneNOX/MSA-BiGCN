import gensim
from gensim.models.keyedvectors import KeyedVectors

# 修改输入的字符串：字母转换为小写，符号/数字与字母分开。返回List[str]
def fixText(raw: str):
    character = 'abcdefghijklmnopqrstuvwxyz '
    raw = raw.lower()
    raw = list(raw)
    i = 0
    for i in range(len(raw)):
        if raw[i] not in character:
            raw[i] = ' ' + raw[i] + ' '
    raw = ''.join(raw)
    return raw
