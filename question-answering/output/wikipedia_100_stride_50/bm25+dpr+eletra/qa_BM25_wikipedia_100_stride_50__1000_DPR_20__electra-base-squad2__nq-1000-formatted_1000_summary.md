### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__nq-1000-formatted_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.398
* BM25 Recall @ 10: 0.475
* BM25 Recall @ 20: 0.564
* BM25 Recall @ 50: 0.656
* BM25 Recall @ 100: 0.706
### Dense Retrieval recall 
* Dense Recall @ 5: 0.592
* Dense Recall @ 10: 0.651
* Dense Recall @ 20: 0.703
* Dense Recall @ 50: 0.757
* Dense Recall @ 100: 0.773
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.157
* BM25 Precision @ 10: 0.135
* BM25 Precision @ 20: 0.118
* BM25 Precision @ 50: 0.097
* BM25 Precision @ 100: 0.083
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.271
* Dense Precision @ 10: 0.231
* Dense Precision @ 20: 0.192
* Dense Precision @ 50: 0.148
* Dense Precision @ 100: 0.118
### F1 
* Mean F1 per q: 0.261
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.383
### Precision 
* Mean precision per q: 0.273
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.4
### Recall 
* Mean recall per q: 0.288
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.415
### Exact Match 
* Mean EM per q: 0.16
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.367
### Time(s) 
* Mean time per q: 19.33s
* Max time per q: 25.9s
* Min time per q: 17.43s
* Std time per q: 0.8s
