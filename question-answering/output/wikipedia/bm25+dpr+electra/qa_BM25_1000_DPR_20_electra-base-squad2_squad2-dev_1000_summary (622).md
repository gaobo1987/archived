### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_squad2-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 251
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.49
* BM25 Recall @ 10: 0.566
* BM25 Recall @ 20: 0.633
* BM25 Recall @ 50: 0.737
* BM25 Recall @ 100: 0.805
### Dense Retrieval recall 
* Dense Recall @ 5: 0.518
* Dense Recall @ 10: 0.594
* Dense Recall @ 20: 0.669
* Dense Recall @ 50: 0.757
* Dense Recall @ 100: 0.817
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.208
* BM25 Precision @ 10: 0.179
* BM25 Precision @ 20: 0.15
* BM25 Precision @ 50: 0.13
* BM25 Precision @ 100: 0.116
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.237
* Dense Precision @ 10: 0.22
* Dense Precision @ 20: 0.194
* Dense Precision @ 50: 0.159
* Dense Precision @ 100: 0.141
### F1 
* Mean F1 per q: 0.216
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.374
### Precision 
* Mean precision per q: 0.22
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.378
### Recall 
* Mean recall per q: 0.228
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.391
### Exact Match 
* Mean EM per q: 0.155
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.362
### Time(s) 
* Mean time per q: 45.2s
* Max time per q: 103.31s
* Min time per q: 22.44s
* Std time per q: 13.68s
