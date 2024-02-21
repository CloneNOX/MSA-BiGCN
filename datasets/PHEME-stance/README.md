### PHEME-rnr-Dataset from:
https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619

新补充的PHEME数据集立场标注，训练集来源于SemEval2017-task8数据集的标注，通过训练立场分类模型以获取整个数据集的立场标签。实验中使用Bin Liang等在 *Zero-Shot Stance Detection via Contrastive Learning* 中提出的PT-HCL模型完成该工作（repo：[BinLiang-NLP/PT-HCL: [WWW 2022] Zero-Shot Stance Detection via Contrastive Learning (github.com)](https://github.com/BinLiang-NLP/PT-HCL)）。

### 完成处理之后的数据文件格式
所有数据集合成一个数据集，在`all/`目录下存储

- `thread_id.txt`：所有source tweet的id，同时用于标识thread

- `post_id.txt`: 所有post的id

- `thread_label.txt`：所有thread的谣言label，已转换成：

    ```text
    'non-rumor': 0, 
    'true': 1, 
    'false': 2, 
    'unverified': 3
    ```

- `structures.json`：传播树结构，以

    ```text
    {
    	thread_id: {
    		post1: {
    			post2:[]
    		}
    		post3: []
    	}
    }
    ```

    的形式存储，值为空列表表示该post是传播树的叶节点。

- `posts.json`：post的文本和发布时间戳，以

    ```text
    {
    	thread_id: {
    		post1: {
    			"time": timestamp
    			"text": text in post
    		}
    		post2: {
    			"time": timestamp
    			"text": text in post
    		}
    	}
    }
    ```


- `posts_label.txt`：所有post的立场label，已转换成

```text
'support': 0,
'query': 1,
'deny': 2,
'comment': 3
```

