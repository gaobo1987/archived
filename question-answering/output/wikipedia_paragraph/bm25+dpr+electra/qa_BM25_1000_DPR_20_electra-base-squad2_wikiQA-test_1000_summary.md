### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_wikiQA-test_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 243
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.025
* BM25 Recall @ 10: 0.041
* BM25 Recall @ 20: 0.049
* BM25 Recall @ 50: 0.07
* BM25 Recall @ 100: 0.082
### Dense Retrieval recall 
* Dense Recall @ 5: 0.066
* Dense Recall @ 10: 0.095
* Dense Recall @ 20: 0.107
* Dense Recall @ 50: 0.119
* Dense Recall @ 100: 0.119
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.006
* BM25 Precision @ 10: 0.005
* BM25 Precision @ 20: 0.003
* BM25 Precision @ 50: 0.001
* BM25 Precision @ 100: 0.001
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.013
* Dense Precision @ 10: 0.01
* Dense Precision @ 20: 0.006
* Dense Precision @ 50: 0.003
* Dense Precision @ 100: 0.001
### F1 
* Mean F1 per q: 0.106
* Median F1 per q: 0.071
* Max F1 per q: 0.8
* Min F1 per q: 0
* Std F1 per q: 0.123
### Precision 
* Mean precision per q: 0.346
* Median precision per q: 0.231
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.369
### Recall 
* Mean recall per q: 0.07
* Median recall per q: 0.045
* Max recall per q: 0.727
* Min recall per q: 0
* Std recall per q: 0.094
### Exact Match 
* Mean EM per q: 0.0
* Median EM per q: 0.0
* Max EM per q: 0.0
* Min EM per q: 0.0
* Std EM per q: 0.0
### Time(s) 
* Mean time per q: 41.61s
* Max time per q: 49.35s
* Min time per q: 37.1s
* Std time per q: 2.67s
