### Pipeline Parameters:
* Name: BM25_DPR_electra-base-squad2_squad2-dev
* BM25 top K = 500
* Dense top K = 10
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1239
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.583
* BM25 Recall @ 10: 0.674
* BM25 Recall @ 20: 0.742
* BM25 Recall @ 50: 0.819
* BM25 Recall @ 100: 0.854
### Dense Retrieval recall 
* Dense Recall @ 5: 0.63
* Dense Recall @ 10: 0.705
* Dense Recall @ 20: 0.762
* Dense Recall @ 50: 0.826
* Dense Recall @ 100: 0.865
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.258 
* BM25 Precision @ 10: 0.229
* BM25 Precision @ 20: 0.206
* BM25 Precision @ 50: 0.173
* BM25 Precision @ 100: 0.153
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.306
* Dense Precision @ 10: 0.271
* Dense Precision @ 20: 0.238
* Dense Precision @ 50: 0.199
* Dense Precision @ 100: 0.175
### F1 
* Avg F1 per q: 0.237
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.384
### Exact Match 
* Avg EM per q: 0.165
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.372
### Time(s) 
* Avg time per q: 29.82s
* Max time per q: 99.42s
* Min time per q: 8.92s
* Std time per q: 12.21s
