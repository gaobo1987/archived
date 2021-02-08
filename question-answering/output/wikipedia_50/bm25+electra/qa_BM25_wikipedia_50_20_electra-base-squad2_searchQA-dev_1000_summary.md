### Pipeline Parameters:
* Name: BM25_wikipedia_50_20_electra-base-squad2_searchQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.492
* BM25 Recall @ 10: 0.575
* BM25 Recall @ 20: 0.633
* BM25 Recall @ 50: 0.633
* BM25 Recall @ 100: 0.633
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.22
* BM25 Precision @ 10: 0.186
* BM25 Precision @ 20: 0.151
* BM25 Precision @ 50: 0.06
* BM25 Precision @ 100: 0.03
### F1 
* Mean F1 per q: 0.135
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.315
### Precision 
* Mean precision per q: 0.129
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.31
### Recall 
* Mean recall per q: 0.155
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.355
### Exact Match 
* Mean EM per q: 0.089
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.285
### Time(s) 
* Mean time per q: 4.25s
* Max time per q: 7.25s
* Min time per q: 0s
* Std time per q: 0.56s
