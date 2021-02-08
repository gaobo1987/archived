### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__triviaQA-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.662
* BM25 Recall @ 10: 0.714
* BM25 Recall @ 20: 0.75
* BM25 Recall @ 50: 0.786
* BM25 Recall @ 100: 0.817
### Dense Retrieval recall 
* Dense Recall @ 5: 0.757
* Dense Recall @ 10: 0.791
* Dense Recall @ 20: 0.813
* Dense Recall @ 50: 0.847
* Dense Recall @ 100: 0.858
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.393
* BM25 Precision @ 10: 0.342
* BM25 Precision @ 20: 0.292
* BM25 Precision @ 50: 0.227
* BM25 Precision @ 100: 0.184
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.478
* Dense Precision @ 10: 0.428
* Dense Precision @ 20: 0.371
* Dense Precision @ 50: 0.296
* Dense Precision @ 100: 0.239
### F1 
* Mean F1 per q: 0.493
* Median F1 per q: 0.5
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.463
### Precision 
* Mean precision per q: 0.494
* Median precision per q: 0.5
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.469
### Recall 
* Mean recall per q: 0.515
* Median recall per q: 0.5
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.476
### Exact Match 
* Mean EM per q: 0.404
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.491
### Time(s) 
* Mean time per q: 18.68s
* Max time per q: 23.14s
* Min time per q: 15.8s
* Std time per q: 0.8s
