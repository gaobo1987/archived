### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__searchQA-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.576
* BM25 Recall @ 10: 0.65
* BM25 Recall @ 20: 0.706
* BM25 Recall @ 50: 0.77
* BM25 Recall @ 100: 0.8
### Dense Retrieval recall 
* Dense Recall @ 5: 0.662
* Dense Recall @ 10: 0.727
* Dense Recall @ 20: 0.774
* Dense Recall @ 50: 0.829
* Dense Recall @ 100: 0.854
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.307
* BM25 Precision @ 10: 0.272
* BM25 Precision @ 20: 0.226
* BM25 Precision @ 50: 0.176
* BM25 Precision @ 100: 0.142
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.368
* Dense Precision @ 10: 0.33
* Dense Precision @ 20: 0.288
* Dense Precision @ 50: 0.232
* Dense Precision @ 100: 0.187
### F1 
* Mean F1 per q: 0.175
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.344
### Precision 
* Mean precision per q: 0.165
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.337
### Recall 
* Mean recall per q: 0.208
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.394
### Exact Match 
* Mean EM per q: 0.11
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.313
### Time(s) 
* Mean time per q: 19.54s
* Max time per q: 22.05s
* Min time per q: 0s
* Std time per q: 0.99s
