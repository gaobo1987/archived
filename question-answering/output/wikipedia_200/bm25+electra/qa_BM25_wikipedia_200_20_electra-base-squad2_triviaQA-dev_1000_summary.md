### Pipeline Parameters:
* Name: BM25_wikipedia_200_20_electra-base-squad2_triviaQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.692
* BM25 Recall @ 10: 0.734
* BM25 Recall @ 20: 0.776
* BM25 Recall @ 50: 0.776
* BM25 Recall @ 100: 0.776
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.394
* BM25 Precision @ 10: 0.337
* BM25 Precision @ 20: 0.283
* BM25 Precision @ 50: 0.113
* BM25 Precision @ 100: 0.057
### F1 
* Mean F1 per q: 0.447
* Median F1 per q: 0.25
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.467
### Precision 
* Mean precision per q: 0.45
* Median precision per q: 0.222
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.473
### Recall 
* Mean recall per q: 0.463
* Median recall per q: 0.25
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.479
### Exact Match 
* Mean EM per q: 0.373
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.484
### Time(s) 
* Mean time per q: 4.3s
* Max time per q: 6.83s
* Min time per q: 3.59s
* Std time per q: 0.52s
