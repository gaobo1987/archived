This folder contains scripts for the evaluation of NER models.

To perform evaluation of NER models on datasets, 
* first go to the [resources](../resources) directory, 
configure your models and datasets.
* then run the following code:
```python
from nlp.ner.evaluation.ner_evaluator import NEREvaluator
lang = 'nl' # or any language you have prepared
NEREvaluator(lang).write_results_to_csv()
```

An example is shown in [example.ipynb](example.ipynb).