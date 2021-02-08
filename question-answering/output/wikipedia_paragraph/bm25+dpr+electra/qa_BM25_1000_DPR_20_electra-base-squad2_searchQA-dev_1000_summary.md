### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_searchQA-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.537
* BM25 Recall @ 10: 0.618
* BM25 Recall @ 20: 0.678
* BM25 Recall @ 50: 0.741
* BM25 Recall @ 100: 0.784
### Dense Retrieval recall 
* Dense Recall @ 5: 0.635
* Dense Recall @ 10: 0.711
* Dense Recall @ 20: 0.763
* Dense Recall @ 50: 0.806
* Dense Recall @ 100: 0.832
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.252
* BM25 Precision @ 10: 0.212
* BM25 Precision @ 20: 0.172
* BM25 Precision @ 50: 0.128
* BM25 Precision @ 100: 0.102
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.337
* Dense Precision @ 10: 0.294
* Dense Precision @ 20: 0.248
* Dense Precision @ 50: 0.187
* Dense Precision @ 100: 0.145
### F1 
* Mean F1 per q: 0.188
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.357
### Precision 
* Mean precision per q: 0.178
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.35
### Recall 
* Mean recall per q: 0.22
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.403
### Exact Match 
* Mean EM per q: 0.123
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.328
### Time(s) 
* Mean time per q: 42.66s
* Max time per q: 48.02s
* Min time per q: 0s
* Std time per q: 2.42s
