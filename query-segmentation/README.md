### 如何使用 How to use this package:

```commandline
pip intall query_segmentation-x.x.x-py3-none-any.whl
```
```python
from qsegmt.nl import NLSegmenter
input_string = 'Koning Willem heeft het Rijksmuseum bezocht.'
output = NLSegmenter().search_segment(input_string)
```
```json
[
   {
      "item":"Koning",
      "lemma":"koning",
      "pos":"NOUN",
      "startOffset":0,
      "endOffset":6,
      "ner":"",
      "isFromUserDict":"false"
   },
   {
      "item":"Willem",
      "lemma":"willem",
      "pos":"NOUN",
      "startOffset":7,
      "endOffset":13,
      "ner":"",
      "isFromUserDict":"false"
   },
   {
      "item":"heeft",
      "lemma":"hebben",
      "pos":"VERB",
      "startOffset":14,
      "endOffset":19,
      "ner":"",
      "isFromUserDict":"false"
   },
   {
      "item":"het",
      "lemma":"het",
      "pos":"DET",
      "startOffset":20,
      "endOffset":23,
      "ner":"",
      "isFromUserDict":"false"
   },
   {
      "item":"Rijksmuseum",
      "lemma":"rijksmuseum",
      "pos":"NOUN",
      "startOffset":24,
      "endOffset":35,
      "ner":[
         {
            "ner":"LOC"
         }
      ],
      "isFromUserDict":"false"
   },
   {
      "item":"bezocht",
      "lemma":"bezoeken",
      "pos":"VERB",
      "startOffset":36,
      "endOffset":43,
      "ner":"",
      "isFromUserDict":"false"
   },
   {
      "item":".",
      "lemma":".",
      "pos":"PUNCT",
      "startOffset":43,
      "endOffset":44,
      "ner":"",
      "isFromUserDict":"false"
   },
   {
      "item":"Koning Willem",
      "lemma":"koning willem",
      "pos":"NOUN",
      "startOffset":0,
      "endOffset":13,
      "ner":[
         {
            "ner":"PER"
         }
      ],
      "isFromUserDict":"false"
   }
]
```
------
### 比较模型 Comparing BERT, Stanza and SpaCy
注: 我们使用了 [bert-base-dutch-cased](https://huggingface.co/wietsedv/bert-base-dutch-cased) 这个模型, 
在[已有数据](qsegmt/nl/data)上细调.

Remark: The BERT variant is [bert-base-dutch-cased](https://huggingface.co/wietsedv/bert-base-dutch-cased), 
find-tuned on the [all training sets](qsegmt/nl/data) combined.

###### 数据 Data description:

|       Data 	| #sentences 	| #tokens/sent. | 
|--------------:|--------------:|--------------:|
| demorgen2000 	|        5,195 	|        13.3 	|
| wikiner 	    |        27,648 |        18.8 	|
| parliamentary |        502 	|        20.9 	|
| meantimenews  |        534 	|        22.0 	|
| europeananews |        602 	|        100.7 	|


###### 延迟 Speed:

|  Speed (ms/sent.)| bert  | stanza | spacy |
|-----------------:|------:|-------:| -----:|
| demorgen2000     | 12.3  | 60.9   | 5.9   |
| wikiner  	       | 12.7  | 73.4   | 7.8   |
| parliamentary    | 11.8  | 77.0   | 8.3   |
| meantimenews     | 12.0  | 79.6   | 8.5   |
| europeananews    | 14.5  | 146.9  | 34.3  |


###### F1 (strict): 
实体边界和种类严格匹配 / exact boundary surface string match and entity type

| F1 (strict)    | bert | stanza | spacy |
|---------------:|-----:| ------:| -----:|
| demorgen2000  | .872  | .873   | .504  |
| wikiner  	    | .911  | .795   | .526  |
| parliamentary | .731  | .657   | .350  |
| meantimenews  | .351  | .213   | .071  |
| europeananews | .581  | .056   | .039  |

###### F1 (exact): 
实体边界严格匹配, 但种类可不同 / exact boundary match over the surface string, regardless of the type

| F1 (exact)    | bert | stanza | spacy |
|---------------:|-----:|------:| -----:|
| demorgen2000  | .947  | .956   | .693  |
| wikiner  	    | .951  | .912   | .736  |
| parliamentary | .807  | .810   | .550  |
| meantimenews  | .367  | .260   | .144  |
| europeananews | .600  | .080   | .069  |

###### F1 (type): 
实体边界和种类可非严格匹配, 但必须有重叠部分 / some overlap between the system tagged entity and the gold annotation is required

| F1 (type)     | bert | stanza | spacy |
|---------------:|-----:|------:| -----:|
| demorgen2000  | .879  | .878   | .516  |
| wikiner  	    | .919  | .808   | .546  |
| parliamentary | .748  | .667   | .357  |
| meantimenews  | .457  | .297   | .117  |
| europeananews | .624  | .077   | .055  |

###### F1 (partial): 

实体边界有重叠部分, 种类可不匹配 / partial boundary match over the surface string, regardless of the type

| F1 (partial)   | bert | stanza | spacy |
|---------------:|-----:|------:| -----:|
| demorgen2000  | .953  | .960   | .706  |
| wikiner  	    | .955  | .921   | .752  |
| parliamentary | .824  | .833   | .567  |
| meantimenews  | .425  | .312   | .192  |
| europeananews | .622  | .095   | .080  |



------
### 如何构建包 How to build the package:

##### 前置条件 Prerequisites:
```commandline
pip install -r requirements.txt
```
此包依赖深度神经网络模型, 因此推荐使用GPU显卡. 详细配置如下.

This package relies on deep neural network models, 
thus a GPU(s) is recommended. 
It also assumes 418+ Nvidia driver and CUDA 10.2+.
Nvidia driver and CUDA compatibility can be found 
[here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

If you have a CUDA other than 10.2+, follow the installation instructions in 
[pytorch.org](https://pytorch.org/), and adjust the PyTorch version during installation.

##### 运行 Run:

```shell script
chmod +x build.sh
./build.sh
```


------
### 使用 Jupyter Notebook, add current venv as kernel

* 添加笔记本核心 Add virtual kernel to Jupyter notebook:

First activate the virtual environment, in the venv:
```
$ pip install ipykernel 
$ python -m ipykernel install --user --name=q-seg-venv
```
Then launch Jupyter notebook, select the kernel 'q-seg-venv'

* 显示核心列表 Show list of Jupiter kernels:
```
$ jupyter kernelspec list
```
* 移除核心 Remove kernel from Jupyter notebook: 
```
$ jupyter kernelspec uninstall unwanted-kernel
```
