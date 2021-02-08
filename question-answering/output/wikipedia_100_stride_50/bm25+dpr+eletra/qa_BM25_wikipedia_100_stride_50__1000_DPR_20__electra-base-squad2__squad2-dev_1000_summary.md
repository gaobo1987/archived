### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__squad2-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.581
* BM25 Recall @ 10: 0.64
* BM25 Recall @ 20: 0.705
* BM25 Recall @ 50: 0.783
* BM25 Recall @ 100: 0.805
### Dense Retrieval recall 
* Dense Recall @ 5: 0.485
* Dense Recall @ 10: 0.573
* Dense Recall @ 20: 0.672
* Dense Recall @ 50: 0.764
* Dense Recall @ 100: 0.808
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.258
* BM25 Precision @ 10: 0.195
* BM25 Precision @ 20: 0.147
* BM25 Precision @ 50: 0.104
* BM25 Precision @ 100: 0.083
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.208
* Dense Precision @ 10: 0.184
* Dense Precision @ 20: 0.158
* Dense Precision @ 50: 0.124
* Dense Precision @ 100: 0.102
### F1 
* Mean F1 per q: 0.349
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.433
### Precision 
* Mean precision per q: 0.358
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.442
### Recall 
* Mean recall per q: 0.366
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.448
### Exact Match 
* Mean EM per q: 0.261
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.439
### Time(s) 
* Mean time per q: 18.99s
* Max time per q: 20.83s
* Min time per q: 14.97s
* Std time per q: 0.64s
