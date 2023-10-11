# BERT-BiGCN & MSA-BiGCN

这个仓库是中山大学2022届本科生毕业论文《基于注意力机制和图卷积神经网络的多任务谣言检测》后续优化的代码实现。原论文的代码实现和baseline运行代码在master分支。

基于以前的工作：

1. 我们改为使用BERT做为编码器，获得词向量和句向量，随后接入BiGCN或Self-Attention以完成谣言检测和立场分类任务。
2. 我们重新整理了数据集，摒弃了旧repo中所有信息塞入JSON文件的做法，改用更规整，更直观的文件存储方式。
3. （TODO）我们将同步调整MSA-BiGCN的代码库，使其适配现在的数据集，采用和现模型相同的训练过程，以便比较。

### 仓库结构

```
MSA-BiGCN
├── BERT-BiGCN
├── MSA-BiGCN
├── README.md
├── datasets
│   ├── PHEME
│   └── semeval2017-task8
├── log // 可选，用于存放训练记录
├── model // 可选，用于存放模型文件
└── requirements.txt
```

### Python库依赖

Python版本`python >= 3.8 `

可通过一下指令安装依赖环境

```shell
$ pip install -r requirements.txt
```

### datasets

dataset文件中包含了[SemEval](https://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools)/[PHEME](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)处理后的数据集。其中PHEME数据集所有数据整合在`datasets/PHEME/all`目录下，SemEval数据集按照原数据集的任务分类分别整理在`data/semeval2017-task8/traindev`和`data/semeval2017-task8/test`内。以SemEval数据集为例，文件含义如下：

```
semeval2017-task8
├── README.md
├── getDataSet.ipynb // 处理数据集的代码
├── process.py // 处理数据集的代码
├── rumorCategory.json // 谣言检测的类别字典
├── stanceCategory.json // 立场分类的类别字典
├── test
│   ├── post_id.txt // 语料中post的id列表，每行存放一个post id
│   ├── post_label.txt // 每个post的立场标签，与post id一一对应，PHEME数据集没有此文件
│   ├── posts.json // 以"post_id: time, text"键值对存储的post信息
│   ├── structures.json // 以"thread_id: structure"键值对存储的thread传播树结构
│   ├── thread_id.txt // 语料中thread的id列表，每行存放一个thread id
│   └── thread_label.txt // 每个thread的立场标签，与thread id一一对应
├── traindev
│   ├── post_id.txt
│   ├── post_label.txt
│   ├── posts.json
│   ├── structures.json
│   ├── thread_id.txt
│   └── thread_label.txt
└── utils.py
```

### BERT-BiGCN

使用huggingface库提供的BERT接口实现嵌入功能的模型，详细更改参考目录下`README.md`。（尚未更新）

### MSA-BiGCN

使用glove词嵌入和Self-Attention实现词和文本的嵌入功能的模型版本，详细更改参考目录下`README.md`。（尚未更新）