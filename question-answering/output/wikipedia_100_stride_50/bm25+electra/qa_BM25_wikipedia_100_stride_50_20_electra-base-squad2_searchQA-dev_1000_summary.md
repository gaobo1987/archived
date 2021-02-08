### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50_20_electra-base-squad2_searchQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.576
* BM25 Recall @ 10: 0.65
* BM25 Recall @ 20: 0.706
* BM25 Recall @ 50: 0.706
* BM25 Recall @ 100: 0.706
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.307
* BM25 Precision @ 10: 0.272
* BM25 Precision @ 20: 0.226
* BM25 Precision @ 50: 0.09
* BM25 Precision @ 100: 0.045
### F1 
* Mean F1 per q: 0.148
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.322
### Precision 
* Mean precision per q: 0.14
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.315
### Recall 
* Mean recall per q: 0.182
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.375
### Exact Match 
* Mean EM per q: 0.093
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.29
### Time(s) 
* Mean time per q: 5.26s
* Max time per q: 7.05s
* Min time per q: 0s
* Std time per q: 0.36s
