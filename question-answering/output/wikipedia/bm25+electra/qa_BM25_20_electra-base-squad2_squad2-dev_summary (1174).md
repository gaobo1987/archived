### Pipeline Parameters:
* Name: BM25_20_electra-base-squad2_squad2-dev
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1174
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.582
* BM25 Recall @ 10: 0.675
* BM25 Recall @ 20: 0.741
* BM25 Recall @ 50: 0.741
* BM25 Recall @ 100: 0.741
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.256
* BM25 Precision @ 10: 0.227
* BM25 Precision @ 20: 0.205
* BM25 Precision @ 50: 0.082
* BM25 Precision @ 100: 0.041
### F1 
* Avg F1 per q: 0.256
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.397
### Exact Match 
* Avg EM per q: 0.187
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.39
### Time(s) 
* Avg time per q: 13.47s
* Max time per q: 44.39s
* Min time per q: 3.62s
* Std time per q: 7.26s
