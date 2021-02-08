### Pipeline Parameters:
* Name: BM25_50_electra-base-squad2_squad2-dev
* BM25 top K = 50
* Reader top K = 1
* USE_GPU = True
* Number of questions: 2321
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.533
* BM25 Recall @ 10: 0.617
* BM25 Recall @ 20: 0.687
* BM25 Recall @ 50: 0.759
* BM25 Recall @ 100: 0.759
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.223
* BM25 Precision @ 10: 0.193
* BM25 Precision @ 20: 0.17
* BM25 Precision @ 50: 0.142
* BM25 Precision @ 100: 0.071
### F1 
* Avg F1 per q: 0.268
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.403
### Exact Match 
* Avg EM per q: 0.198
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.398
### Time(s) 
* Avg time per q: 31.98s
* Max time per q: 137.04s
* Min time per q: 9.58s
* Std time per q: 16.15s
