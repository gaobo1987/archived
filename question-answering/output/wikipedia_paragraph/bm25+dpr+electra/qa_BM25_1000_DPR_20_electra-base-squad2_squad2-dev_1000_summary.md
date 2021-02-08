### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_squad2-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.499
* BM25 Recall @ 10: 0.573
* BM25 Recall @ 20: 0.634
* BM25 Recall @ 50: 0.708
* BM25 Recall @ 100: 0.752
### Dense Retrieval recall 
* Dense Recall @ 5: 0.467
* Dense Recall @ 10: 0.56
* Dense Recall @ 20: 0.646
* Dense Recall @ 50: 0.744
* Dense Recall @ 100: 0.797
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.17
* BM25 Precision @ 10: 0.13
* BM25 Precision @ 20: 0.098
* BM25 Precision @ 50: 0.073
* BM25 Precision @ 100: 0.06
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.184
* Dense Precision @ 10: 0.157
* Dense Precision @ 20: 0.129
* Dense Precision @ 50: 0.099
* Dense Precision @ 100: 0.08
### F1 
* Mean F1 per q: 0.319
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.427
### Precision 
* Mean precision per q: 0.323
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.432
### Recall 
* Mean recall per q: 0.331
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.441
### Exact Match 
* Mean EM per q: 0.243
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.429
### Time(s) 
* Mean time per q: 41.38s
* Max time per q: 46.58s
* Min time per q: 37.08s
* Std time per q: 1.43s
