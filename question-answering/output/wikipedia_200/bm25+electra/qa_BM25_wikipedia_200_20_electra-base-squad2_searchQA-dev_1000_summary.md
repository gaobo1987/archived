### Pipeline Parameters:
* Name: BM25_wikipedia_200_20_electra-base-squad2_searchQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.632
* BM25 Recall @ 10: 0.698
* BM25 Recall @ 20: 0.755
* BM25 Recall @ 50: 0.755
* BM25 Recall @ 100: 0.755
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.324
* BM25 Precision @ 10: 0.276
* BM25 Precision @ 20: 0.229
* BM25 Precision @ 50: 0.092
* BM25 Precision @ 100: 0.046
### F1 
* Mean F1 per q: 0.139
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.315
### Precision 
* Mean precision per q: 0.129
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.304
### Recall 
* Mean recall per q: 0.17
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.367
### Exact Match 
* Mean EM per q: 0.089
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.285
### Time(s) 
* Mean time per q: 4.15s
* Max time per q: 6.25s
* Min time per q: 0s
* Std time per q: 0.45s
