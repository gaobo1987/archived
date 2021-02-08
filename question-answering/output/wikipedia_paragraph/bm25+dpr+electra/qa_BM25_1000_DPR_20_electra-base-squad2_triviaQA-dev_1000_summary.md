### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_triviaQA-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.607
* BM25 Recall @ 10: 0.667
* BM25 Recall @ 20: 0.708
* BM25 Recall @ 50: 0.762
* BM25 Recall @ 100: 0.8
### Dense Retrieval recall 
* Dense Recall @ 5: 0.734
* Dense Recall @ 10: 0.772
* Dense Recall @ 20: 0.803
* Dense Recall @ 50: 0.828
* Dense Recall @ 100: 0.848
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.317
* BM25 Precision @ 10: 0.264
* BM25 Precision @ 20: 0.216
* BM25 Precision @ 50: 0.164
* BM25 Precision @ 100: 0.129
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.444
* Dense Precision @ 10: 0.384
* Dense Precision @ 20: 0.325
* Dense Precision @ 50: 0.244
* Dense Precision @ 100: 0.186
### F1 
* Mean F1 per q: 0.457
* Median F1 per q: 0.4
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.462
### Precision 
* Mean precision per q: 0.46
* Median precision per q: 0.333
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.469
### Recall 
* Mean recall per q: 0.475
* Median recall per q: 0.4
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.475
### Exact Match 
* Mean EM per q: 0.368
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.482
### Time(s) 
* Mean time per q: 42.29s
* Max time per q: 51.22s
* Min time per q: 38.0s
* Std time per q: 1.68s
