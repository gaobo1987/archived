This directory contains neural reader models.

#### Dense Passage Retriever (DPR)
- [paper](https://arxiv.org/pdf/2004.04906.pdf)
- [download reference](https://github.com/facebookresearch/DPR/blob/master/data/download_data.py)

Datasets used:
* Natural Questions (NQ)
* TriviaQA 
* WebQuestions 
* CuratedTREC 
* SQuAD

Single-trained Retriever:
- Bi-encoder weights trained on NQ data and HF bert-base-uncased model.
- [Download](https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp)

Multi-trained Retriever:
- Bi-encoder weights trained on multi set data (except SQuAD) and HF bert-base-uncased model.
- [Download](https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp)
