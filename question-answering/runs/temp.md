[[_TOC_]]

### SQuAD-v1.1-dev.json
#### subset: 100, seed: 42

Abbreviation: 
* aq = all questions
* aqo = answered questions only
* F1 = Average F1 Score
* EM = Average Exact Match Score
* RM = Average Regex Match Score
* FSP = Fixed Sized Passage

| Method (retriever/reader) | top k retriever | avg inference time per question (s) | answered questions | F1 (aq) | F1 (aqo) | EM (aq) | EM (aqo) | RM (aq) | RM (aqo) |
|---------------------------|-----------------|-------------------|--------------------|---------|----------|---------|----------|---------|----------|
| BM25 / BM25               | 10              |                  |                    | 0.059   | 0.059    |         |          |         |          |
| BM25 / RoBERTa-squad      | 10              |                  |                    | 0.129   | 0.175    |         |          |         |          |
| BM25 / RoBERTa-squad     | 20 | 13.0 | 97.4 | 0.168 | 0.181 | 0.110 | 0.118 | 0.100 | 0.108 |
| BM25 / Electra-squad      | 20              | 10.1                 | 97%                | 0.259   | 0.267    | 0.210   | 0.216    | 0.210   | 0.216    |
| BM25 / BERT-squad         | 20              |                  | 99%                | 0.224   | 0.226    | 0.150   | 0.152    | 0.130   | 0.131    |
| BM25/ spanBERT-squadv2 | 20 | 18.5 | 100.0 | 0.201 | 0.201 | 0.170 | 0.170 | 0.160 | 0.160 |


#### subset: 1000, seed: 42

| Method (retriever/reader)                                    | top k retriever | avg inference time per question (s) | answered questions† | F1 (aq)   | F1 (aqo)  | EM (aq)   | EM (aqo)  | RM (aq)   | RM (aqo)  |
|--------------------------------------------------------------|-----------------|-------------------------------------|---------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Reader selection**                                         |                 |                                     |                     |           |           |           |           |           |           |
| BM25/ bm25                                                   | 20              | 2.3                                 | ~~100.0~~           | 0.060     | 0.060     | 0.000     | 0.000     | 0.011     | 0.011     |
| BM25/ bert-base-uncased-squad2                               | 20              | 9.5                                 | ~~99.9~~            | 0.219     | 0.219     | 0.162     | 0.162     | 0.152     | 0.152     |
| BM25/ roberta-base-squad2                                    | 20              | 13.1                                | ~~98.0~~            | 0.214     | 0.227     | 0.149     | 0.158     | 0.137     | 0.145     |
| BM25/ electra-base-squad2                                    | 20              | 10.2                                | ~~98.8~~            | **0.259** | **0.268** | **0.193** | **0.200** | 0.186     | **0.192** |
| BM25/ span-bert-squadv2                                      | 20              | 19.0                                | ~~100.0~~           | 0.213     | 0.213     | 0.158     | 0.158     | 0.162     | 0.162     |
| BM25/ electra_large_discriminator_squad2_512                 | 20              | 18.1                                | ~~100.0~~           | 0.244     | 0.244     | 0.192     | 0.192     | **0.191** | 0.191     |
| BM25 / bert-large-uncased-whole-word-masking-finetuned-squad | 20              | 19.5                                | ~~100.0~~           | 0.220     | 0.220     | 0.160     | 0.160     | 0.159     | 0.159     |
| **Index selection**                                          |                 |                                     |                     |           |           |           |           |           |           |
| paragraph BM25*/ electra-base-squad2                         | 20              | ~~7.4~~ (CPU)                       | ~~97.9~~            | 0.341     | 0.362     | 0.274     | 0.291     | 0.254     | 0.270     |
| paragraph BM25**/ electra-base-squad2                        | 20              | ~~8.1~~ (CPU)                       | ~~94.2~~            | 0.292     | 0.340     | 0.226     | 0.263     | 0.219     | 0.255     |
| paragraph BM25*** / electra-base-squad2                      | 20              | ~~8.0~~ (CPU)                       | ~~97.8~~            | 0.354     | 0.376     | 0.286     | 0.304     | 0.265     | 0.282     |
| FSP 100 BM25 / electra-base-squad2                           | 20              | 3.4                                 | ~~97.6~~            | 0.375     | 0.401     | 0.300     | 0.321     | 0.289     | 0.309     |
| FSP 100 + 50 stride BM25 / electra-base-squad2               | 20              | 3.4                                 | 99.5                | **0.422** | **0.442** | **0.344** | **0.360** | **0.324** | **0.339** |
| FSP 200 BM25 / electra-base-squad2                           | 20              | 3.5                                 | ~~97.9~~            | 0.397     | 0.421     | 0.324     | 0.344     | 0.307     | 0.326     |
| **Additional components**                                    |                 |                                     |                     |           |           |           |           |           |           |
| FSP 100 + 50 stride BM25 / electra-base-squad2 + scorer (mu = 1, so no performance gain!)| 20      | 5.6             | 95.5                | 0.422     | 0.442     | 0.344     | 0.360     | 0.324     | 0.339     |
| FSP 100 + 50 stride BM25 / DPR (1000) / electra-base-squad2  | 20              | 21                                  | 96.4                | 0.373     | 0.387     | 0.293     | 0.304     | NA        | NA        |

- \* Index is made by splitting the articles on double new-line character (`\n\n`) only.
- ** Similar as above, but the first paragraph (== title of article) and empty paragraphs are removed further the search fields for ES are both the title and text instead of only text.
- *** Same index as ** but this time the only search field is the text, title is ignored.
- † There was a bug in the determination of the fraction of answered questions. The cells with the strike-through still used the old way, without strike-through are correct values.

### naturalQuestions-dev-squad.json
#### subset: 1000, seed: 42

| Method (retriever/reader)                                            | top k retriever | avg inference time per question (s) | answered questions† | F1 (aq)   | F1 (aqo)  | EM (aq)   | EM (aqo)  | RM (aq)   | RM (aqo)  |
|----------------------------------------------------------------------|-----------------|-------------------------------------|---------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Reader selection**                                                 |                 |                                     |                     |           |           |           |           |           |           |
| BM25/ bert-base-uncased-squad2 | 20 | 7.8 | 99.8 | 0.167 | 0.172 | 0.097 | 0.100 | 0.093 | 0.096 |
| BM25 / roberta-base-squad2 | 20 | 10.3 | 99.0 | 0.156 | 0.175 | 0.082 | 0.092 | 0.082 | 0.092 |
| BM25 / span-bert-squadv2 | 20 | 15.4 | **100.0** | **0.184** | **0.184** | **0.112** | **0.112** | **0.116** | **0.116** |
| BM25 / electra-base-squad2 | 20 | 9.2 | 99.8 | 0.165 | 0.170 | 0.089 | 0.092 | 0.104 | 0.107 |
| **Index selection**                                                  |                 |                                     |                     |           |           |           |           |           |           |
| FSP 50 BM25/ electra-base-squad2                                     | 20              | 3.4                                 | ~~97.0~~            | 0.189     | 0.206     | 0.108     | 0.117     | 0.112     | 0.122     |
| FSP 100 BM25 / electra-base-squad2                                   | 20              | 3.8                                 | ~~97.8~~            | 0.194     | 0.207     | **0.110** | **0.117** | 0.119     | 0.127     |
| FSP 100 + 50 stride BM25 / electra-base-squad2                       | 20              | 6.7                                 | ~~97.5~~            | 0.197     | **0.212** | 0.107     | 0.115     | **0.120** | **0.129** |
| FSP 150 BM25/ electra-base-squad2                                    | 20              | 4.1                                 | ~~97.7~~            | **0.198** | 0.211     | 0.109     | 0.116     | 0.114     | 0.122     |
| *FSP 100 + 50 stride BM25 / span-bert-squadv2* | 20 | 4.8 | 100.0 | 0.196 | 0.196 | 0.112 | 0.112 | 0.122 | 0.122 |
| FSP 200 BM25 / electra-base-squad2                                   | 20              | 3.8                                 | ~~98.1~~            | 0.194     | 0.205     | 0.107     | 0.113     | 0.116     | 0.122     |
| paragraph BM25 / electra-base-squad2                                 | 20              | 3.6                                 | ~~97.4~~            | 0.179     | 0.193     | 0.104     | 0.112     | 0.112     | 0.121     |
| **Additional components**                                            |                 |                                     |                     |           |           |           |           |           |           |
| FSP 100 + 50 stride BM25 / electra-base-squad2 + Scorer (mu = 0.988) | 20              | 2.1                                 | ~~99.3~~            | **0.200** | **0.215** | **0.111** | **0.119** | **0.123** | **0.132** |
| FSP 100 + 50 stride BM25 / DPR (1000) / electra-base-squad2          | 20              | 28.8                                | 94.9                |    0.261  | **0.275** |   0.160   | **0.169** | NA        | NA        |
| FSP 100 + 50 stride BM25 / DPR (1000) / electra-base-squad2 + Scorer (mu=0.55)| 20     | 28.8                                | 94.9                | **0.292** | NA        | **0.181** | NA        | NA        | NA        |



## Results on other datasets
#### wikipedia_100_stride_50
Note that the NQ and SQuAD subsets are different from the ones used in the experiments above (though with about 20-30% overlap)

* Retrieval

|        seed=42, excl.questions w/o answers |       NQ-dev (1000)      |     QuasarT-dev (1000)   |     SearchQA-dev (1000)  |      SQuAD2-dev (1000)   |     TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|-------------------------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:-------------------------:|
|                     BM25  Retrieval Recall | @5=.37 @10=.46 @20=.55   | @5=.44 @10=.49 @20=.56   | @5=.58 @10=.65 @20=.71   |**@5=.58 @10=.64 @20=.71**| @5=.66 @10=.61 @20=.75   | @5=.02 @10=.04 @20=.05    |
|     BM25 + DPR (top 1000) Retrieval Recall |**@5=.58 @10=.64 @20=.70**|**@5=.56 @10=.61 @20=.66**|**@5=.66 @10=.73 @20=.77**| @5=.49 @10=.57 @20=.67   |**@5=.76 @10=.79 @20=.81**|**@5=.09 @10=.10 @20=.10** |
|                  BM25  Retrieval Precision | @5=.16 @10=.14 @20=.12   | @5=.22 @10=.19 @20=.17   | @5=.31 @10=.27 @20=.23   |**@5=.26 @10=.20 @20=.15**| @5=.39 @10=.34 @20=.29   | @5=.005 @10=.004 @20=.002 |
| BM25 + DPR (top 1000)  Retrieval Precision |**@5=.26 @10=.22 @20=.19**|**@5=.33 @10=.29 @20=.25**|**@5=.37 @10=.33 @20=.29**| @5=.21 @10=.18 @20=.16   |**@5=.48 @10=.43 @20=.37**|**@5=.020 @10=.010 @20=.005** |


* Reading

|         seed=42, excl.questions w/o answers              |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|---------------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                               BM25 + Electra (top 20) F1 | mean=.202, std=.350     | mean=.233 std=.390     | mean=.148, std=.322    |**mean=.421 std=.455**  | mean=.455 std=.465     | mean=.084 std=.093        |
|              BM25 + DPR (top 1000) + Electra (top 20) F1 | mean=.248, std=.375     | mean=.272 std=.412     | mean=.175, std=.344    | mean=.349 std=.433     | mean=.493 std=.463     | mean=.109 std=.117        |
| BM25 + DPR (top 1000) + Electra + BERTserini (top 20) F1 | **mean=.287, mu=.45**   | **mean=.291, mu=.60**  |**mean=.178, mu=.49**   | mean=.350 mu=.74       | **mean=.514 mu=.5**    | **mean=.126 mu=.42**      |
|                               BM25 + Electra (top 20) EM | mean=.124, std=.330     | mean=.175 std=.380     | mean=.093, std=.290    |**mean=.333 std=.471**  | mean=.375 std=.484     | mean=0 std=.0             |
|              BM25 + DPR (top 1000) + Electra (top 20) EM | mean=.155, std=.362     | mean=.207 std=.405     | mean=.110, std=.313    | mean=.261 std=.439     | mean=.404 std=.491     | mean=0 std=.0             |
| BM25 + DPR (top 1000) + Electra + BERTserini (top 20) EM | **mean=.181, mu=.45**   | **mean=.217, mu=.60**  |**mean=.113, mu=.75**   | mean=.263 mu=.92       | **mean=.412 mu=.36**   | **mean=.0 mu=None**       |
