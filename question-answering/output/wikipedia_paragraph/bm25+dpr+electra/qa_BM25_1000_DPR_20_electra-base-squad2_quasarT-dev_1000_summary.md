### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_quasarT-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.396
* BM25 Recall @ 10: 0.474
* BM25 Recall @ 20: 0.536
* BM25 Recall @ 50: 0.607
* BM25 Recall @ 100: 0.648
### Dense Retrieval recall 
* Dense Recall @ 5: 0.55
* Dense Recall @ 10: 0.616
* Dense Recall @ 20: 0.646
* Dense Recall @ 50: 0.693
* Dense Recall @ 100: 0.717
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.175
* BM25 Precision @ 10: 0.151
* BM25 Precision @ 20: 0.126
* BM25 Precision @ 50: 0.1
* BM25 Precision @ 100: 0.082
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.292
* Dense Precision @ 10: 0.255
* Dense Precision @ 20: 0.216
* Dense Precision @ 50: 0.163
* Dense Precision @ 100: 0.127
### F1 
* Mean F1 per q: 0.278
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.417
### Precision 
* Mean precision per q: 0.272
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.417
### Recall 
* Mean recall per q: 0.304
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.444
### Exact Match 
* Mean EM per q: 0.211
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.408
### Time(s) 
* Mean time per q: 42.58s
* Max time per q: 50.47s
* Min time per q: 38.43s
* Std time per q: 1.7s
