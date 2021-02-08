### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_naturalQuestions-dev-clean_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.326
* BM25 Recall @ 10: 0.403
* BM25 Recall @ 20: 0.486
* BM25 Recall @ 50: 0.58
* BM25 Recall @ 100: 0.637
### Dense Retrieval recall 
* Dense Recall @ 5: 0.554
* Dense Recall @ 10: 0.617
* Dense Recall @ 20: 0.67
* Dense Recall @ 50: 0.703
* Dense Recall @ 100: 0.733
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.119
* BM25 Precision @ 10: 0.102
* BM25 Precision @ 20: 0.087
* BM25 Precision @ 50: 0.071
* BM25 Precision @ 100: 0.06
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.236
* Dense Precision @ 10: 0.193
* Dense Precision @ 20: 0.155
* Dense Precision @ 50: 0.114
* Dense Precision @ 100: 0.09
### F1 
* Mean F1 per q: 0.236
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.365
### Precision 
* Mean precision per q: 0.252
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.385
### Recall 
* Mean recall per q: 0.254
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.391
### Exact Match 
* Mean EM per q: 0.136
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.343
### Time(s) 
* Mean time per q: 40.61s
* Max time per q: 45.29s
* Min time per q: 34.82s
* Std time per q: 1.52s
