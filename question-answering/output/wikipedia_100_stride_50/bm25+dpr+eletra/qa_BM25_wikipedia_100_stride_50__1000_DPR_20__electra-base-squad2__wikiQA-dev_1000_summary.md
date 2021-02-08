### Pipeline Parameters:
* Name: BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__wikiQA-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 20
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 126
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.024
* BM25 Recall @ 10: 0.04
* BM25 Recall @ 20: 0.048
* BM25 Recall @ 50: 0.071
* BM25 Recall @ 100: 0.087
### Dense Retrieval recall 
* Dense Recall @ 5: 0.087
* Dense Recall @ 10: 0.095
* Dense Recall @ 20: 0.095
* Dense Recall @ 50: 0.103
* Dense Recall @ 100: 0.103
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.005
* BM25 Precision @ 10: 0.004
* BM25 Precision @ 20: 0.002
* BM25 Precision @ 50: 0.002
* BM25 Precision @ 100: 0.001
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.017
* Dense Precision @ 10: 0.01
* Dense Precision @ 20: 0.005
* Dense Precision @ 50: 0.003
* Dense Precision @ 100: 0.001
### F1 
* Mean F1 per q: 0.109
* Median F1 per q: 0.08
* Max F1 per q: 0.48
* Min F1 per q: 0
* Std F1 per q: 0.117
### Precision 
* Mean precision per q: 0.355
* Median precision per q: 0.293
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.355
### Recall 
* Mean recall per q: 0.073
* Median recall per q: 0.047
* Max recall per q: 0.375
* Min recall per q: 0
* Std recall per q: 0.086
### Exact Match 
* Mean EM per q: 0.0
* Median EM per q: 0.0
* Max EM per q: 0.0
* Min EM per q: 0.0
* Std EM per q: 0.0
### Time(s) 
* Mean time per q: 19.02s
* Max time per q: 20.48s
* Min time per q: 17.42s
* Std time per q: 0.53s
