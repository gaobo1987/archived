### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_squad2-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 96
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.396
* BM25 Recall @ 10: 0.521
* BM25 Recall @ 20: 0.604
* BM25 Recall @ 50: 0.656
* BM25 Recall @ 100: 0.719
### Dense Retrieval recall 
* Dense Recall @ 5: 0.562
* Dense Recall @ 10: 0.625
* Dense Recall @ 20: 0.667
* Dense Recall @ 50: 0.729
* Dense Recall @ 100: 0.781
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.188
* BM25 Precision @ 10: 0.171
* BM25 Precision @ 20: 0.164
* BM25 Precision @ 50: 0.14
* BM25 Precision @ 100: 0.123
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.265
* Dense Precision @ 10: 0.222
* Dense Precision @ 20: 0.196
* Dense Precision @ 50: 0.167
* Dense Precision @ 100: 0.15
### F1 
* Mean F1 per q: 0.219
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.363
### Precision 
* Mean precision per q: 0.226
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.378
### Recall 
* Mean recall per q: 0.226
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.374
### Exact Match 
* Mean EM per q: 0.146
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.353
### Time(s) 
* Mean time per q: 47.06s
* Max time per q: 128.62s
* Min time per q: 26.74s
* Std time per q: 15.05s
