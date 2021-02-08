### Pipeline Parameters:
* Name: BM25_wikipedia_150_20_electra-base-squad2_triviaQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.667
* BM25 Recall @ 10: 0.722
* BM25 Recall @ 20: 0.763
* BM25 Recall @ 50: 0.763
* BM25 Recall @ 100: 0.763
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.375
* BM25 Precision @ 10: 0.32
* BM25 Precision @ 20: 0.269
* BM25 Precision @ 50: 0.108
* BM25 Precision @ 100: 0.054
### F1 
* Mean F1 per q: 0.435
* Median F1 per q: 0.114
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.463
### Precision 
* Mean precision per q: 0.438
* Median precision per q: 0.111
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.47
### Recall 
* Mean recall per q: 0.449
* Median recall per q: 0.068
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.476
### Exact Match 
* Mean EM per q: 0.356
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.479
### Time(s) 
* Mean time per q: 4.23s
* Max time per q: 7.35s
* Min time per q: 3.56s
* Std time per q: 0.5s
