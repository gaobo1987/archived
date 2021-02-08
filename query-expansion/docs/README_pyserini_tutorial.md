# Pyserini tutorial

### What is Pyserini
[Pyserini](https://github.com/castorini/pyserini) 
provides a simple python interface to the Java-based 
Information Retrieval (IR) toolkit Anserini. [Anserini](https://github.com/castorini/anserini) is an 
open-source IR toolkit built on [Lucene](https://lucene.apache.org/).

### How to install Pyserini
```shell script
pip install pyserini
```

### What is MS MARCO
[MS MARCO](https://microsoft.github.io/msmarco/) is a 
[Human GEnerated MAchine Reading COmprehension](https://arxiv.org/abs/1611.09268) 
dataset, which is used for IR research and search engine evaluation.

* msmarco-doc dataset:
    - for document ranking tasks (full/re-ranking)
    - 3.2 million documents
    - [download link](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz)
    - [reference link](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-doc.md)

* msmarco-passage dataset:
    - for passage ranking tasks (full/re-ranking)
    - 8.8 million passages
    - [download link](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz)
    - [reference link](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md)
   
    
### How to build pyserini index on msmarco data
...

