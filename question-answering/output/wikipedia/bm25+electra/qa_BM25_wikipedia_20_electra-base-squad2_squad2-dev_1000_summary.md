### Pipeline Parameters:
* Name: <BM25_wikipedia>_20_<electra-base-squad2>_squad2-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.493
* BM25 Recall @ 10: 0.582
* BM25 Recall @ 20: 0.66
* BM25 Recall @ 50: 0.66
* BM25 Recall @ 100: 0.66
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.211
* BM25 Precision @ 10: 0.188
* BM25 Precision @ 20: 0.164
* BM25 Precision @ 50: 0.066
* BM25 Precision @ 100: 0.033
### F1 
* Mean F1 per q: 0.223
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.374
### Precision 
* Mean precision per q: 0.229
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.381
### Recall 
* Mean recall per q: 0.235
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.391
### Exact Match 
* Mean EM per q: 0.153
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.36
### Time(s) 
* Mean time per q: 13.29s
* Max time per q: 58.97s
* Min time per q: 4.2s
* Std time per q: 8.36s
