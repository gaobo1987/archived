This folder contains the original data [ner.jsonl](ner.jsonl), 
and the four datasets derived from it, [all.txt](all.txt), 
[train.txt](train.txt), [test.txt](test.txt), [dev.txt](dev.txt), 
with their NER labels kept as is, namely

```json
{
  "O": 72663,
  "B-ORG": 615,
  "I-ORG": 441,
  "B-PERSON": 637,
  "I-PERSON": 387,
  "B-PRODUCT": 67,
  "I-PRODUCT": 61,
  "B-GPE": 836,
  "I-GPE": 85,
  "B-LOC": 27,
  "I-LOC": 12,
  "B-EVENT": 56,
  "I-EVENT": 62
}
```

More info about the original jsonl data :

It is from a [project](https://github.com/eellak/gsoc2018-spacy) 
that integrates the Greek language into spaCy. 

The related datasets can be found 
[here](https://github.com/eellak/gsoc2018-spacy/tree/dev/spacy/lang/el/training/datasets/annotated_data).
