### Instructions for evaluation

*  Data format

An good example is the 
[eval.json](data/nl/eval.json), 
of which a snippet is shown below. It is a list of json objects 
that contain manually curated queries, and their "gold" and "pred" segmentations.
"gold" refers to the ground-truth, and "pred" refers to model-predictions. 
The evaluation functions rely on this format to perform evaluation.

```json
[{
      "query":"hoofdstad van Suriname",
      "gold":[
         {
            "item":"hoofdstad",
            "lemma":"hoofdstad",
            "pos":"NOUN",
            "startOffSet":0,
            "endOffSet":9,
            "ner":"",
            "isMinimumToken":true
         },
         {
            "item":"van",
            "lemma":"van",
            "pos":"ADP",
            "startOffSet":10,
            "endOffSet":13,
            "ner":"",
            "isMinimumToken":true
         },
         {
            "item":"Suriname",
            "lemma":"suriname",
            "pos":"NOUN",
            "startOffSet":14,
            "endOffSet":22,
            "ner":[
               {
                  "ner":"LOC"
               }
            ],
            "isMinimumToken":true
         }
      ],
      "pred":[
         {
            "item":"hoofdstad",
            "lemma":"hoofdstad",
            "pos":"NOUN",
            "startOffSet":0,
            "endOffSet":9,
            "ner":"",
            "isMinimumToken":true,
            "isStopWord":false
         },
         {
            "item":"van",
            "lemma":"van",
            "pos":"ADP",
            "startOffSet":10,
            "endOffSet":13,
            "ner":"",
            "isMinimumToken":true,
            "isStopWord":true
         },
         {
            "item":"Suriname",
            "lemma":"suriname",
            "pos":"NOUN",
            "startOffSet":14,
            "endOffSet":22,
            "ner":"",
            "isMinimumToken":true,
            "isStopWord":false
         }
      ]
}]
```


* How to run

Import the evaluate function and point it to the preprocessed evaluation dataset.
The 'key' parameter indicates which aspect of the model you want to evaluate, 
currently supported values are: "pos", "lemma" and "ner".

```python
from nlp.evaluation.utils import evaluate
output = evaluate('path/to/eval_data.json', key='pos')
```

If the key is 'pos' or 'lemma', the output contains 
average and weighted F,P,R scores, 
as well as F,P,R scores per individual category, for example:

```json
{
  "weighted_f1": 0.8837909114784446,
  "weighted_precision": 0.8941123696292743,
  "weighted_recall": 0.883495145631068,
  "average_f1": 0.7368448851789988,
  "average_precision": 0.734965367394616,
  "average_recall": 0.7472483122146705,
  "f1s": ["..."],
  "precisions": ["..."],
  "recalls": ["..."]
}
```

If the key is 'ner', 
it outputs detailed F,P,R scores in four modes for NER evaluation, 
overall and per NER label, along with other informatoin.

An example can be found in [nlp/evaluation/run.ipynb](example.ipynb).
