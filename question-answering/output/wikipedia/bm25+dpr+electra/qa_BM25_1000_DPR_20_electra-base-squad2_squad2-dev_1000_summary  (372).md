### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_squad2-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 374
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.468
* BM25 Recall @ 10: 0.561
* BM25 Recall @ 20: 0.642
* BM25 Recall @ 50: 0.743
* BM25 Recall @ 100: 0.791
### Dense Retrieval recall 
* Dense Recall @ 5: 0.535
* Dense Recall @ 10: 0.588
* Dense Recall @ 20: 0.663
* Dense Recall @ 50: 0.735
* Dense Recall @ 100: 0.797
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.194
* BM25 Precision @ 10: 0.173
* BM25 Precision @ 20: 0.151
* BM25 Precision @ 50: 0.125
* BM25 Precision @ 100: 0.111
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.235
* Dense Precision @ 10: 0.207
* Dense Precision @ 20: 0.183
* Dense Precision @ 50: 0.152
* Dense Precision @ 100: 0.131
### F1 
* Mean F1 per q: 0.214
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.366
### Precision 
* Mean precision per q: 0.218
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.373
### Recall 
* Mean recall per q: 0.223
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.379
### Exact Match 
* Mean EM per q: 0.144
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.351
### Time(s) 
* Mean time per q: 47.78s
* Max time per q: 98.88s
* Min time per q: 23.5s
* Std time per q: 13.55s
