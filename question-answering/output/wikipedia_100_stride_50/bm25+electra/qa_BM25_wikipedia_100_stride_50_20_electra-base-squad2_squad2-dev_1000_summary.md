### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50_20_electra-base-squad2_squad2-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.581
* BM25 Recall @ 10: 0.64
* BM25 Recall @ 20: 0.705
* BM25 Recall @ 50: 0.705
* BM25 Recall @ 100: 0.705
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.258
* BM25 Precision @ 10: 0.195
* BM25 Precision @ 20: 0.147
* BM25 Precision @ 50: 0.059
* BM25 Precision @ 100: 0.029
### F1 
* Mean F1 per q: 0.421
* Median F1 per q: 0.174
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.455
### Precision 
* Mean precision per q: 0.429
* Median precision per q: 0.183
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.461
### Recall 
* Mean recall per q: 0.438
* Median recall per q: 0.177
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.469
### Exact Match 
* Mean EM per q: 0.333
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.471
### Time(s) 
* Mean time per q: 5.19s
* Max time per q: 6.58s
* Min time per q: 4.31s
* Std time per q: 0.21s
