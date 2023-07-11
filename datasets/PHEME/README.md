### PHEME-rnr-Dataset from:
https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619

### 完成处理之后的数据文件格式

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

    
