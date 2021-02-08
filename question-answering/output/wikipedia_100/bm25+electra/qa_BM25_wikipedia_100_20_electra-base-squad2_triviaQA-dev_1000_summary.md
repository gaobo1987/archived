### Pipeline Parameters:
* Name: BM25_wikipedia_100_20_electra-base-squad2_triviaQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.65
* BM25 Recall @ 10: 0.708
* BM25 Recall @ 20: 0.743
* BM25 Recall @ 50: 0.743
* BM25 Recall @ 100: 0.743
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.355
* BM25 Precision @ 10: 0.303
* BM25 Precision @ 20: 0.249
* BM25 Precision @ 50: 0.1
* BM25 Precision @ 100: 0.05
### F1 
* Mean F1 per q: 0.44
* Median F1 per q: 0.2
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.463
### Precision 
* Mean precision per q: 0.443
* Median precision per q: 0.183
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.469
### Recall 
* Mean recall per q: 0.454
* Median recall per q: 0.2
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.476
### Exact Match 
* Mean EM per q: 0.362
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.481
### Time(s) 
* Mean time per q: 4.28s
* Max time per q: 7.2s
* Min time per q: 3.56s
* Std time per q: 0.53s
