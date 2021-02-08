### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_squad2-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 38
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.553
* BM25 Recall @ 10: 0.684
* BM25 Recall @ 20: 0.789
* BM25 Recall @ 50: 0.921
* BM25 Recall @ 100: 0.974
### Dense Retrieval recall 
* Dense Recall @ 5: 0.632
* Dense Recall @ 10: 0.711
* Dense Recall @ 20: 0.789
* Dense Recall @ 50: 0.868
* Dense Recall @ 100: 0.921
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.211
* BM25 Precision @ 10: 0.197
* BM25 Precision @ 20: 0.166
* BM25 Precision @ 50: 0.136
* BM25 Precision @ 100: 0.123
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.289
* Dense Precision @ 10: 0.255
* Dense Precision @ 20: 0.224
* Dense Precision @ 50: 0.175
* Dense Precision @ 100: 0.158
### F1 
* Mean F1 per q: 0.197
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.35
### Precision 
* Mean precision per q: 0.217
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.377
### Recall 
* Mean recall per q: 0.204
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.366
### Exact Match 
* Mean EM per q: 0.105
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.307
### Time(s) 
* Mean time per q: 49.15s
* Max time per q: 96.49s
* Min time per q: 31.74s
* Std time per q: 15.82s
