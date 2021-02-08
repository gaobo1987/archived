# Performance Summary
This folder contains run scripts, which run full pipelines to 
answer open domain questions.

------
## Performance across indexes


------
## Performance on individual indexes
#### wikipedia

* number of tokens per document:
    - mean = 372.3
    - std = 778.5
    - min = 1
    - max = 62,206
    - median = 143.0

* Retrieval

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25  Retrieval Recall | @5=.57 @10=.64 @20=.71  | @5=.55 @10=.60 @20=.65 | @5=.70 @10=.75 @20=.80 | @5=.49 @10=.58 @20=.66 | @5=.73 @10=.77 @20=.81 | @5=.09 @10=.10 @20=.11    |
|                   BM25  Retrieval Precision | @5=.24 @10=.20 @20=.17  | @5=.28 @10=.24 @20=.21 | @5=.38 @10=.32 @20=.26 | @5=.21 @10=.19 @20=.16 | @5=.42 @10=.37 @20=.31 | @5=.019 @10=.011 @20=.006 |


* Reading

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25 + Electra (top 20) F1 | mean=.172 std=.324      | mean=.235 std=.392     | mean=.147 std=.321     | mean=.223 std=.374     | mean=.419 std=.464     | mean=.073 std=.102        |
|                  BM25 + Electra (top 20) EM | mean=.102 std=.302      | mean=.176 std=.381     | mean=.095 std=.293     | mean=.153 std=.360     | mean=.347 std=.476     | mean=0 std=.0             |

  
#### wikipedia_paragraph

* number of tokens per paragraph:
    - mean = 61.1
    - std = 52.6
    - min = 1
    - max = 5503
    - median = 48

* Retrieval

|             seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25para  Retrieval Recall | @5=.33 @10=.40 @20=.49  | @5=.40 @10=.48 @20=.54 | @5=.54 @10=.62 @20=.68 |**@5=.50 @10=.57 @20=.63**| @5=.61 @10=.67 @20=.71 | @5=.01 @10=.02 @20=.04    |
|      BM25para + DPR (top 1000) Retrieval Recall |**@5=.55 @10=.62 @20=.67**|**@5=.55 @10=.62 @20=.65**|**@5=.64 @10=.71 @20=.76**|@5=.47 @10=.56 @20=.65|**@5=.73 @10=.77 @20=.80**|**@5=.09 @10=.11 @20=.11**|
|                   BM25para  Retrieval Precision | @5=.12 @10=.10 @20=.09  | @5=.18 @10=.15 @20=.13 | @5=.25 @10=.21 @20=.17 | @5=.17 @10=.13 @20=.10 | @5=.32 @10=.26 @20=.22 | @5=.002 @10=.002 @20=.002 |
|  BM25para + DPR (top 1000)  Retrieval Precision |**@5=.24 @10=.19 @20=.16**|**@5=.29 @10=.26 @20=.22**|**@5=.34 @10=.29 @20=.25**|**@5=.18 @10=.16 @20=.13**|**@5=.44 @10=.38 @20=.33**|**@5=.02 @10=.01 @20=.007**|


* Reading

|             seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25para + Electra (top 20) F1 | mean=.192 std=.335      | mean=.236 std=.397     | mean=.142 std=.321     |**mean=.358 std=.443**  | mean=.424 std=.462     | mean=.087 std=.109        |
| BM25para + DPR (top 1000) + Electra (top 20) F1 |**mean=.236 std=.365**   |**mean=.278 std=.417**  |**mean=.188 std=.357**  | mean=.319 std=.427     |**mean=.457 std=.462**  |**mean=.095 std=.106**     |
|                  BM25para + Electra (top 20) EM | mean=.105 std=.307      | mean=.185 std=.388     | mean=.095 std=.293     |**mean=.281 std=.449**  | mean=.345 std=.475     | mean=0 std=.0             |
| BM25para + DPR (top 1000) + Electra (top 20) EM |**mean=.136 std=.343**   |**mean=.211 std=.408**  |**mean=.123 std=.328**  | mean=.243 std=.429     |**mean=.368 std=.482**  | mean=0 std=.0             |


#### wikipedia_200
* Retrieval

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25  Retrieval Recall | @5=.46 @10=.54 @20=.60  | @5=.48 @10=.54 @20=.60 | @5=.63 @10=.70 @20=.76 | @5=.58 @10=.66 @20=.71 | @5=.69 @10=.73 @20=.78 |                           |
|                   BM25  Retrieval Precision | @5=.18 @10=.15 @20=.13  | @5=.23 @10=.21 @20=.17 | @5=.32 @10=.28 @20=.23 | @5=.22 @10=.17 @20=.14 | @5=.39 @10=.34 @20=.28 |                           |

* Reading

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25 + Electra (top 20) F1 | mean=.185 std=.325      | mean=.230 std=.385     | mean=.139 std=.315     | mean=.383 std=.449     | mean=.447 std=.467     |                           |
|                  BM25 + Electra (top 20) EM | mean=.096 std=.295      | mean=.166 std=.372     | mean=.089 std=.285     | mean=.306 std=.461     | mean=.373 std=.484     |                           |


#### wikipedia_150
* Retrieval

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25  Retrieval Recall | @5=.43 @10=.52 @20=.59  | @5=.46 @10=.52 @20=.58 | @5=.61 @10=.68 @20=.72 | @5=.55 @10=.61 @20=.68 | @5=.67 @10=.72 @20=.76 |                           |
|                   BM25  Retrieval Precision | @5=.16 @10=.14 @20=.12  | @5=.22 @10=.20 @20=.16 | @5=.31 @10=.26 @20=.21 | @5=.19 @10=.15 @20=.12 | @5=.38 @10=.32 @20=.27 |                           |

* Reading

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25 + Electra (top 20) F1 | mean=.211 std=.349      | mean=.229 std=.386     | mean=.137 std=.313     | mean=.372 std=.448     | mean=.435 std=.463     |                           |
|                  BM25 + Electra (top 20) EM | mean=.121 std=.326      | mean=.169 std=.375     | mean=.087 std=.282     | mean=.297 std=.457     | mean=.356 std=.479     |                           |


#### wikipedia_100_stride_50
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


#### wikipedia_100
* Retrieval

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25  Retrieval Recall | @5=.40 @10=.48 @20=.56  | @5=.44 @10=.50 @20=.56 | @5=.59 @10=.65 @20=.71 | @5=.56 @10=.63 @20=.68 | @5=.65 @10=.71 @20=.74 | @5=.024 @10=.040 @20=.056 |
|                   BM25  Retrieval Precision | @5=.15 @10=.12 @20=.10  | @5=.20 @10=.18 @20=.15 | @5=.28 @10=.23 @20=.19 | @5=.19 @10=.15 @20=.11 | @5=.36 @10=.30 @20=.25 | @5=.005 @10=.004 @20=.003 |

* Reading

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25 + Electra (top 20) F1 | mean=.204 std=.347      | mean=.228 std=.390     | mean=.135 std=.311     | mean=.381 std=.447     | mean=.440 std=.463     | mean=.095 std=.116        |
|                  BM25 + Electra (top 20) EM | mean=.120 std=.325      | mean=.175 std=.380     | mean=.083 std=.276     | mean=.298 std=.457     | mean=.362 std=.481     | mean=0 std=.0             |

#### wikipedia_50
* Retrieval

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25  Retrieval Recall | @5=.32 @10=.39 @20=.46  | @5=.38 @10=.45 @20=.50 | @5=.49 @10=.58 @20=.63 | @5=.48 @10=.55 @20=.60 | @5=.58 @10=.64 @20=.71 | @5=.024 @10=.040 @20=.056 |
|                   BM25  Retrieval Precision | @5=.11 @10=.10 @20=.08  | @5=.17 @10=.15 @20=.12 | @5=.22 @10=.19 @20=.15 | @5=.19 @10=.12 @20=.09 | @5=.30 @10=.25 @20=.20 | @5=.005 @10=.004 @20=.003 |

* Reading

|         seed=42, excl.questions w/o answers |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|--------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25 + Electra (top 20) F1 | mean=.193 std=.340      | mean=.229 std=.391     | mean=.135 std=.315     | mean=.331 std=.433     | mean=.415 std=.462     |                           |
|                  BM25 + Electra (top 20) EM | mean=.113 std=.317      | mean=.174 std=.379     | mean=.089 std=.285     | mean=.262 std=.440     | mean=.342 std=.474     |                           |



### Data Summary
|  excl.questions w/o answers |                                NQ-dev (excl. long answers)                               |                                        QuasarT-dev                                        |                                  SearchQA-dev                                  |                                     SQuAD2-dev                                    |                                       TriviaQA-dev                                       |                                      WikiQA-dev                                      |
|----------------------------:|:----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
|                  #questions |                                                                                    4,201 |                                                                                     3,000 |                                                                         21,613 |                                                                             5,928 |                                                                                   11,313 |                                                                                  126 |
|         #answers p.question |                                                                                        2 |                                                                                         1 |                                                                              1 |                                                                               1.7 |                                                                  15 (variants / aliases) |                                                                                  1.1 |
|          #tokens p.question |                                                                                        9 |                                                                                        11 |                                                                             15 |                                                                                10 |                                                                                       14 |                                                                                    6 |
|            #tokens p.answer |                                                                                      4.2 |                                                                                         2 |                                                                              2 |                                                                               3.6 |                                                                                      2.3 |                                                                                   26 |
|                        6W1H | who=33.2%, what=16.5%, why=0.5%, when17.1%, where=12.6%, which=2%, how=5.0% others=15.2% | who=10.7%, what=32.4%, why=0.1%, when=0.8%, where=2.1%, which=12.8%, how=1.8%, others=53% | who=0%, what=0.1%, why=0%, when=0.6%, where=0%, which=0%, how=0.1%, others=99% | who=8%, what=49%, why=1.6%, when=6.4%, where=3.8%, which=4%, how=9.7%, others=17% | who=9.7%, what=21.2%, why=0.1%, when=0.4%, where=1%, which=22.6%, how=1.4%, others=43.7% | who=11.9%, what=57.9%, why=0%, when=8.7%, where=13.5%, which=0%, how=7.9%, others=0% |

Remarks:
* NQ (NaturalQuestions): real web queries accompanied with wikipedia articles, 
from which people annotate short/long answers, or yes/no answers.
    - "who wrote the song photograph by ringo starr"
    - "when was the first hunger games book published"
    - "to aru kagaku no railgun s episode 3"
    - "dominant alleles are always the most common allele in a population"
    - "original cast of natasha pierre and the great comet of 1812"
* QuasarT: factoid/trivia questions
    - "Lockjaw is another name for which disease"
    - "Which vegetable is a Welsh emblem ?"
    - "Where did judy garland 's family have their vaudeville act"
    - "What nationality was Oddjob"
* SearchQA: Jeopardy question-answer pairs (factoid/trivia, but "none-questions")
    - "The bestselling passenger car of all time is this company's Corolla"
    - "Like a door, a Broadway show does these 2 things"
    - "One of the 2 main types of chemical bonds"
* SQuAD2: Questions are created by people after given a Wikipedia article
    - "What German ruler invited Huguenot immigration?"
    - "What percentage of electrical power in the United States is made by steam turbines?"
    - "What do nuclear power plants heat to create electricity?"
* TriviaQA: factoid/trivia questions
    - "Which Lloyd Webber musical premiered in the US on 10th December 1993?"
    - "In Greek mythology, who were Arges, Brontes and Steropes?"
    - "Who first hosted Family Feud?"
* WikiQA: questions sampled from Bing query logs (factoid, WH-questions, but with long answers)
    - "how big is bmc software in houston?"
    - "what does a plus-minus sign mean"
    - "what kind of literature did john steinbeck writing"
    
    
### Installation notes:
Install haystack locally:

```shell script
cd haystack-0.3.0
pip install --editable .
```
