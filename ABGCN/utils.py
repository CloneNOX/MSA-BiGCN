def flattenStructure(structure: dict):
    Ids = []
    if not structure:
        return Ids # 当前dict为空，直接返回空列表
    Ids += list(structure.keys())
    for id in structure:
        if structure[id]: # 子dict不为空，递归地展开
            Ids += flattenStructure(structure[id])
    return Ids
    