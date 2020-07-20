### 如何使用 How to use this package:

前往 [OneBox](https://onebox.huawei.com/p/b4c213f39001495699fba98bba09f4a9) 下载 wheel 安装文件。

Go to [OneBox](https://onebox.huawei.com/p/b4c213f39001495699fba98bba09f4a9) and download the wheel package.

```commandline
pip intall query_segmentation-x.x.x-py3-none-any.whl
```
```python
from qsegmt.nl import NLSegmenter
input_string = 'Koning Willem heeft het Rijksmuseum bezocht.'
output = NLSegmenter().segment(input_string)
```
```json
[
   {
      "item":"Koning",
      "lemma":"koning",
      "pos":"NOUN",
      "startOffSet":0,
      "endOffSet":6,
      "ner":"",
      "isMinimumToken":true,
      "isStopWord":false
   },
   {
      "item":"Willem",
      "lemma":"willem",
      "pos":"NOUN",
      "startOffSet":7,
      "endOffSet":13,
      "ner":"",
      "isMinimumToken":true,
      "isStopWord":false
   },
   {
      "item":"heeft",
      "lemma":"hebben",
      "pos":"VERB",
      "startOffSet":14,
      "endOffSet":19,
      "ner":"",
      "isMinimumToken":true,
      "isStopWord":true
   },
   {
      "item":"het",
      "lemma":"het",
      "pos":"DET",
      "startOffSet":20,
      "endOffSet":23,
      "ner":"",
      "isMinimumToken":true,
      "isStopWord":true
   },
   {
      "item":"Rijksmuseum",
      "lemma":"rijksmuseum",
      "pos":"NOUN",
      "startOffSet":24,
      "endOffSet":35,
      "ner":[
         {
            "ner":"LOC"
         }
      ],
      "isMinimumToken":true,
      "isStopWord":false
   },
   {
      "item":"bezocht",
      "lemma":"bezoeken",
      "pos":"VERB",
      "startOffSet":36,
      "endOffSet":43,
      "ner":"",
      "isMinimumToken":true,
      "isStopWord":false
   },
   {
      "item":".",
      "lemma":".",
      "pos":"PUNCT",
      "startOffSet":43,
      "endOffSet":44,
      "ner":"",
      "isMinimumToken":true,
      "isStopWord":false
   },
   {
      "item":"Koning Willem",
      "lemma":"koning willem",
      "pos":"NOUN",
      "startOffSet":0,
      "endOffSet":13,
      "ner":[
         {
            "ner":"PER"
         }
      ],
      "isMinimumToken":false,
      "isStopWord":false
   }
]
```

------
### 如何构建包 How to build the package:

##### 前置条件 Prerequisites:
```commandline
pip install -r requirements.txt --no-cache-dir 
```
* 此包依赖深度神经网络模型, 因此推荐使用GPU显卡，详细配置如下. 
注意匹配 Nvidia 418+ 显卡，以及 CUDA 10.2+，如果版本不符，
请参考 [pytorch.org](https://pytorch.org/get-started/locally/) 安装对应的 PyTorch.

* 如果训练本地定制的spaCy模型，为加速训练速度，根据本机 CUDA 版本，
安装 [cupy](https://docs-cupy.chainer.org/en/stable/install.html).

* This package relies on deep neural network models, 
thus a GPU(s) is recommended. 
It also assumes 418+ Nvidia driver and CUDA 10.2+.
Nvidia driver and CUDA compatibility can be found 
[here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).
If you have a CUDA other than 10.2+, follow the installation instructions in 
[pytorch.org](https://pytorch.org/get-started/locally/), and adjust the PyTorch version during installation.

* If you want to train your own spaCy model, 
to accelerate training, please follow the instructions 
to install [cupy](https://docs-cupy.chainer.org/en/stable/install.html).

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
$ python -m ipykernel install --user --name=qsegmt-venv
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

### BERT models

##### BERT
First introduced in 2019, Google’s BERT (Bidirectional Encoder Representations from Transformers) 
is a powerful and popular language representation model 
designed to pre-train deep bidirectional representations from unlabeled text. 
Studies show that BERT models trained on a single language 
notably outperform the multilingual version. 

- [blog](https://syncedreview.com/2020/01/23/hallo-hallo-ku-leuven-tu-berlin-introduce-robbert-a-sota-dutch-bert/)
- [paper](https://arxiv.org/abs/1810.04805)
- [huggingface pretrained models](https://huggingface.co/transformers/pretrained_models.html)

------

##### RoBERTa
The improved version of BERT, introduced in 2019 
by researchers from Facebook AI and University of Washington, Seattle.

- [fairseq roberta examples](https://github.com/pytorch/fairseq/tree/master/examples/roberta)

------

##### RobBERT
Unlike previous approaches that have used earlier implementations of BERT 
to train a Dutch-language BERT, the new research uses RoBERTa. 
RobBERT was pre-trained on 6.6 billion words totaling 39 GB of text 
from the Dutch section of the OSCAR corpus.

- [blog](https://syncedreview.com/2020/01/23/hallo-hallo-ku-leuven-tu-berlin-introduce-robbert-a-sota-dutch-bert/)
- [huggingface transformers model](https://huggingface.co/pdelobelle/robBERT-base)
- [author page](https://people.cs.kuleuven.be/~pieter.delobelle/)
- [paper](https://arxiv.org/abs/2001.06286)

------