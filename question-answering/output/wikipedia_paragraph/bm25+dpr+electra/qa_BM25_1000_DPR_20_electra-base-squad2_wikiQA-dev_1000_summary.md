### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_wikiQA-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 126
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.008
* BM25 Recall @ 10: 0.024
* BM25 Recall @ 20: 0.04
* BM25 Recall @ 50: 0.063
* BM25 Recall @ 100: 0.087
### Dense Retrieval recall 
* Dense Recall @ 5: 0.087
* Dense Recall @ 10: 0.111
* Dense Recall @ 20: 0.111
* Dense Recall @ 50: 0.111
* Dense Recall @ 100: 0.119
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.002
* BM25 Precision @ 10: 0.002
* BM25 Precision @ 20: 0.002
* BM25 Precision @ 50: 0.002
* BM25 Precision @ 100: 0.001
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.017
* Dense Precision @ 10: 0.012
* Dense Precision @ 20: 0.007
* Dense Precision @ 50: 0.003
* Dense Precision @ 100: 0.002
### F1 
* Mean F1 per q: 0.095
* Median F1 per q: 0.077
* Max F1 per q: 0.45
* Min F1 per q: 0
* Std F1 per q: 0.106
### Precision 
* Mean precision per q: 0.362
* Median precision per q: 0.286
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.387
### Recall 
* Mean recall per q: 0.06
* Median recall per q: 0.042
* Max recall per q: 0.3
* Min recall per q: 0
* Std recall per q: 0.07
### Exact Match 
* Mean EM per q: 0.0
* Median EM per q: 0.0
* Max EM per q: 0.0
* Min EM per q: 0.0
* Std EM per q: 0.0
### Time(s) 
* Mean time per q: 40.93s
* Max time per q: 44.62s
* Min time per q: 37.93s
* Std time per q: 1.22s
