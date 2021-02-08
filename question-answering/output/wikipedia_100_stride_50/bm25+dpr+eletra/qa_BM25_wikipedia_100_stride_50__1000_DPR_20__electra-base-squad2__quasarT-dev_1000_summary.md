### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__quasarT-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.438
* BM25 Recall @ 10: 0.492
* BM25 Recall @ 20: 0.562
* BM25 Recall @ 50: 0.627
* BM25 Recall @ 100: 0.676
### Dense Retrieval recall 
* Dense Recall @ 5: 0.56
* Dense Recall @ 10: 0.607
* Dense Recall @ 20: 0.656
* Dense Recall @ 50: 0.706
* Dense Recall @ 100: 0.725
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.218
* BM25 Precision @ 10: 0.19
* BM25 Precision @ 20: 0.171
* BM25 Precision @ 50: 0.139
* BM25 Precision @ 100: 0.116
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.325
* Dense Precision @ 10: 0.286
* Dense Precision @ 20: 0.247
* Dense Precision @ 50: 0.199
* Dense Precision @ 100: 0.163
### F1 
* Mean F1 per q: 0.272
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.412
### Precision 
* Mean precision per q: 0.267
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.414
### Recall 
* Mean recall per q: 0.301
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.44
### Exact Match 
* Mean EM per q: 0.207
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.405
### Time(s) 
* Mean time per q: 19.25s
* Max time per q: 21.23s
* Min time per q: 17.39s
* Std time per q: 0.57s
