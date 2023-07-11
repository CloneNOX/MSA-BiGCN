def flattenStructure(structure: dict):
    '''
    展开字典形式存储的传播树结构, 获取所有节点的id
    Input:
        - structure: dict
    Output:
        - Ids: list
    '''
    Ids = []
    if not structure:
        return Ids # 当前dict为空，直接返回空列表
    Ids += list(structure.keys())
    for id in structure:
        if structure[id]: # 子dict不为空，递归地展开
            Ids += flattenStructure(structure[id])
    return Ids
    