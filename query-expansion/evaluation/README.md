#### How to run trev_eval:
Specify your qrels gold data and your run file 
that contains your own search results:

```shell script
java -jar evaluation/jtreceval-0.0.5-jar-with-dependencies.jar qrels.txt run.txt
```
e.g. 
```shell script
java -jar evaluation/jtreceval-0.0.5-jar-with-dependencies.jar evaluation/qrels.msmarco-doc.dev.txt runs/run-msmarco-doc-bm25-qe5.txt
```

#### qrels file format:
The format of a qrels file is as follows:

<pre>
TOPIC      ITERATION      DOCUMENT#      RELEVANCY
</pre>

* **TOPIC** is the topic number,
* **ITERATION** is the feedback iteration 
(almost always zero and not used),
* **DOCUMENT#** is the official document number 
that corresponds to the "docno" field in the documents, 
* **RELEVANCY** is a binary code, 
0 for not relevant and 1 for relevant.

See more details [here](https://trec.nist.gov/data/qrels_eng/).

#### trec_eval result format:
| Field |                                 Description                               |
|------: | :--------------------------------------------------------------------------|
| runid | Name of the run (is the name given on the last field of the results file) |
| num_q | Total number of evaluated queries |
| num_ret | Total number of retrieved documents |
| num_rel | Total number of relevant documents (according to the qrels file) |
| num_rel_ret | Total number of relevant documents retrieved (in the results file) |
| map | Mean average precision (map) |
| gm_map | Average precision. Geometric mean |
| Rprec | Precision of the first R documents, where R are the number os relevants |
| bpref | Binary preference |
| recip_rank | Reciprical Rank |
| iprec_at_recall_0.00 | Interpolated Recall - Precision Averages at 0.00 recall |
| iprec_at_recall_0.10 | Interpolated Recall - Precision Averages at 0.10 recall |
| iprec_at_recall_0.20 | Interpolated Recall - Precision Averages at 0.20 recall |
| iprec_at_recall_0.30 | Interpolated Recall - Precision Averages at 0.30 recall |
| iprec_at_recall_0.40 | Interpolated Recall - Precision Averages at 0.40 recall |
| iprec_at_recall_0.50 | Interpolated Recall - Precision Averages at 0.50 recall |
| iprec_at_recall_0.60 | Interpolated Recall - Precision Averages at 0.60 recall |
| iprec_at_recall_0.70 | Interpolated Recall - Precision Averages at 0.70 recall |
| iprec_at_recall_0.80 | Interpolated Recall - Precision Averages at 0.80 recall |
| iprec_at_recall_0.90 | Interpolated Recall - Precision Averages at 0.90 recall |
| iprec_at_recall_1.00 | Interpolated Recall - Precision Averages at 1.00 recall |
| P_5 | Precision of the 5 first documents |
| P_10 | Precision of the 10 first documents |
| P_15 | Precision of the 15 first documents |
| P_20 | Precision of the 20 first documents |
| P_30 | Precision of the 30 first documents |
| P_100 | Precision of the 100 first documents |
| P_200 | Precision of the 200 first documents |
| P_500 | Precision of the 500 first documents |
| P_1000 | Precision of the 1000 first documents |

See more details in this [blog](http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system).
