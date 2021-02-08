### Pipeline Parameters:
* Name: BM25_wikipedia_50_20_electra-base-squad2_triviaQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.575
* BM25 Recall @ 10: 0.644
* BM25 Recall @ 20: 0.706
* BM25 Recall @ 50: 0.706
* BM25 Recall @ 100: 0.706
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.295
* BM25 Precision @ 10: 0.246
* BM25 Precision @ 20: 0.203
* BM25 Precision @ 50: 0.081
* BM25 Precision @ 100: 0.041
### F1 
* Mean F1 per q: 0.415
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.462
### Precision 
* Mean precision per q: 0.423
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.471
### Recall 
* Mean recall per q: 0.426
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.473
### Exact Match 
* Mean EM per q: 0.342
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.474
### Time(s) 
* Mean time per q: 4.32s
* Max time per q: 8.99s
* Min time per q: 3.56s
* Std time per q: 0.65s
