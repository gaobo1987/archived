### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__naturalQuestions-dev-clean_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.371
* BM25 Recall @ 10: 0.458
* BM25 Recall @ 20: 0.554
* BM25 Recall @ 50: 0.637
* BM25 Recall @ 100: 0.68
### Dense Retrieval recall 
* Dense Recall @ 5: 0.582
* Dense Recall @ 10: 0.64
* Dense Recall @ 20: 0.697
* Dense Recall @ 50: 0.741
* Dense Recall @ 100: 0.762
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.155
* BM25 Precision @ 10: 0.136
* BM25 Precision @ 20: 0.118
* BM25 Precision @ 50: 0.097
* BM25 Precision @ 100: 0.082
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.264
* Dense Precision @ 10: 0.221
* Dense Precision @ 20: 0.185
* Dense Precision @ 50: 0.141
* Dense Precision @ 100: 0.114
### F1 
* Mean F1 per q: 0.248
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.375
### Precision 
* Mean precision per q: 0.265
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.397
### Recall 
* Mean recall per q: 0.272
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.405
### Exact Match 
* Mean EM per q: 0.155
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.362
### Time(s) 
* Mean time per q: 18.99s
* Max time per q: 20.56s
* Min time per q: 17.29s
* Std time per q: 0.51s
