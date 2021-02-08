### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__squad-1000-formatted_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.57
* BM25 Recall @ 10: 0.625
* BM25 Recall @ 20: 0.697
* BM25 Recall @ 50: 0.768
* BM25 Recall @ 100: 0.811
### Dense Retrieval recall 
* Dense Recall @ 5: 0.484
* Dense Recall @ 10: 0.572
* Dense Recall @ 20: 0.678
* Dense Recall @ 50: 0.77
* Dense Recall @ 100: 0.829
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.252
* BM25 Precision @ 10: 0.189
* BM25 Precision @ 20: 0.146
* BM25 Precision @ 50: 0.106
* BM25 Precision @ 100: 0.084
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.213
* Dense Precision @ 10: 0.185
* Dense Precision @ 20: 0.161
* Dense Precision @ 50: 0.124
* Dense Precision @ 100: 0.102
### F1 
* Mean F1 per q: 0.373
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.445
### Precision 
* Mean precision per q: 0.381
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.451
### Recall 
* Mean recall per q: 0.386
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.456
### Exact Match 
* Mean EM per q: 0.293
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.455
### Time(s) 
* Mean time per q: 21.12s
* Max time per q: 33.63s
* Min time per q: 14.99s
* Std time per q: 4.11s
