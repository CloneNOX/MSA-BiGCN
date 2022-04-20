# MSA-BiGCN/MSA-BiGCN-rws

这个仓库是中山大学2022届本科生毕业论文《基于注意力机制和图卷积神经网络的多任务谣言检测》代码实现及baseline代码汇总。

仓库结构如下：

MSA-BiGCN：

- BiGCN：BiGCN模型的开源代码及原论文
- DualHierarchicalTransformer： CHT/HT模型的开源代码及原论文
- MSA-BiGCN：MSA-BiGCN/MSA-BiGCN-rws模型的实现代码
- MT：MTUS/MTES模型的实现代码及原论文
- dataset：实验中用到的处理好的数据集，包括SemEval和PHEME



### Python库依赖

Python版本`python 3.8.12`

使用到的库：

```
tqdm 4.62.3
numpy 1.21.2
pytorch 1.10.2
scikit-learn 1.0.2
torch_geometric 2.0.3
```



### dataset

dataset文件中包含了[SemEval]([Data and Tools < SemEval-2017 Task 8 (qcri.org)](https://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools))/[PHEME]([PHEME dataset for Rumour Detection and Veracity Classification (figshare.com)](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078))处理后的数据集，以Json文件形式存储。若希望重新生成数据集，需要将数据集源文件下载到对应目录下，解压，并运行`getDataset.ipynb`中的所有单元格。（SemEvalRemake数据集是SemEval数据集的重新划分，在本文实验中并没有使用到）

**注意：**本数据集中不包含glove预训练词向量文件。正确运行模型代码需要确保`dataset/glove/`目录存在。并且在目录中下载**glove.twitter.27B**[源文件](https://nlp.stanford.edu/projects/glove/)，并运行`fix.py`。



### MSA-BiGCN/MSA-BiGCN-rws

`MSABiGCN.py`中包含了模型的实现在文件，`train.py`是多任务学习的训练代码，`trainOnlyRumor.py`是仅进行谣言检测单任务学习的代码。

如果要运行多任务谣言检测请执行

```shell
python train.py --dataPath ../dataset/semeval2017-task8/ --logName ./log/final/needNoStance/orign/25w2v-64s2v-64hidden-0.2dropout-0.001000weight.txt --savePath ./model/final/needNoStance/orign/25w2v-64s2v-64hidden-0.2dropout-0.001000weight.pt --device cuda:0
```

`--dataPath`是必须的参数，指定使用哪一个数据集，其余默认参数请参考源代码。`--device`参数可以指定运行设备。`--logName`和`--savePath`分别是保存记录和模型的路径，如果使用默认参数，请保证已建立`log/`和`model/`目录。`--needStance`参数决定了是否使用立场分类任务到谣言检测任务的连接，默认值是True，使用值False可以取消连接。

`trainOnlyRumor.py`代码可以运行单任务谣言检测的模型训练，参数和`train.py`是相似的。

运行`runNeedStance.sh`、`compareNeedStance.sh`、`compareNeedNoStance.sh`、`runOnlyRumorPHEME.sh`和`runOnlyRumorSemEval.sh`脚本可以进行网格搜索。若需筛选参数，可以使用`result.ipynb`中提供的代码。

`log_copy/`目录中给出了运行实验中的一些实验结果，以Json形式保存，将目录更名为`log`可直接使用`result.ipynb`画图。



### MT

`model`中包含了模型的实现在文件，`train.py`是多任务学习的训练代码，`trainOnlyRumor.py`是仅进行谣言检测单任务学习的代码。使用的参数和MSA-BiGCN是相似的，详细的默认参数请参考源代码。

运行`run`脚本将得到MTUS/MTES模型的实验结果。



### BiGCN

本仓库的实验环境是兼容BiGCN的实验的，其余请参考BiGCN模型中的`readme.md`文件。

我们适配SemEval数据集和PHEME数据集到BiGCN模型中，通过运行`data/SemEval/`和`data/PHEME/`中`getset.ipynb`中的所有单元格，可以获得适配BiGCN数据处理源代码的数据集文件。

在BiGCN目录下，终端运行：

```shell
python ./Process/getSemevalgraph.py
python ./Process/getPHEMEgraph.py
```

处理数据集。

运行：

```shell
python ./model/SemEval/BiGCN_SemEval.py 100
python ./model/PHEME/BiGCN_PHEME.py 100
```

在实验用的两个数据集中训练BiGCN模型。



### DualHierarchicalTransformer

DHT模型已经包含了适配的SemEval数据集和PHEME数据集，运行方法请参考目录下`README.md`文件。