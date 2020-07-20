### NER 模型比较 Comparing BERT, Stanza and SpaCy
注: 我们使用了 [bert-base-dutch-cased](https://huggingface.co/wietsedv/bert-base-dutch-cased) 这个模型, 
在[已有数据](data/)上细调. 
实体分类包括: B-LOC, I-LOC, B-PER, I-PER, B-ORG, I-ORG, B-MISC, I-MISC 和 O.

Remark: The BERT variant is [bert-base-dutch-cased](https://huggingface.co/wietsedv/bert-base-dutch-cased), 
find-tuned on the [all training sets](data/) combined.
The NER tags include: B-LOC, I-LOC, B-PER, I-PER, B-ORG, I-ORG, B-MISC, I-MISC and O.


###### 数据 Data description:

|       Data 	| #sentences 	| #tokens/sent. | 
|--------------:|--------------:|--------------:|
| demorgen2000 	|        5,195 	|        13.3 	|
| wikiner 	    |        27,648 |        18.8 	|
| parliamentary |        502 	|        20.9 	|
| meantimenews  |        534 	|        22.0 	|
| europeananews |        602 	|        100.7 	|


###### 延迟 Speed:
One sentence per batch;
One thread;
One 8-core CPU (3.60GHz), 16G RAM;
One Nvidia GTX 1070 GPU, 8G VM;

|  Speed (ms/sent.)| bert  | stanza | spacy |
|-----------------:|------:|-------:| -----:|
| demorgen2000     | 11.5  | 25.4   | 1.8   |
| wikiner  	       | 12.6  | 34.7   | 2.0   |
| parliamentary    | 11.9  | 37.4   | 2.1   |
| meantimenews     | 11.9  | 40.1   | 2.1   |
| europeananews    | 14.4  | 78.7   | 5.2   |

###### NER 效果评估:
我们可以根据四种模式的 F1, Precision, Recall 来测评模型表现，这四种模式为：
* strict： 实体边界和种类严格匹配
* exact： 实体边界严格匹配, 但种类可不同
* type： 实体边界和种类可非严格匹配, 但必须有重叠部分
* partial: 实体边界有重叠部分, 种类可不匹配 

在此仅列出strict模式下的 F1 分数，全部结果参见 [evaluation_results_test.tsv](evaluation_results_test.tsv).

We can evaluate the performance of the models 
in terms of F1, Precision and Recall in four modes,
* strict: exact boundary surface string match and entity type 
* exact: exact boundary match over the surface string, regardless of the type
* type: some overlap between the system tagged entity and the gold annotation is required
* and partial： partial boundary match over the surface string, regardless of the type

Here only lists the strict F1 scores, for comprehensive evaluation, see [evaluation_results_test.tsv](evaluation_results_test.tsv).

| | demorgen2000 | europeananews | meantimenews | parliamentary | wikiner |
|---:| ---:| ---: | ---: | ---: | ---: |
| spacy_custom_default | 0.380 | 0.064 | 0.129 | 0.405 | 0.494 |
| spacy_custom_blank | 0.392 | 0.053 | 0.124 | 0.417 | 0.496 |
| spacy_default | 0.496 | 0.037 | 0.071 | 0.351 | 0.525 |
| stanza_default | 0.874 | 0.055 | 0.217 | 0.659 | 0.794 |
| bert_base_dutch_cased_ft | 0.872 | 0.581 | 0.351 | 0.731 | 0.911 |
| bert_base_dutch_cased_uncased_ft | 0.759 | 0.439 | 0.270 | 0.547 | 0.841 |
| bert_base_dutch_cased_ft_uncased_ft | 0.775 | 0.490 | 0.337 | 0.600 | 0.850 |
| bert_base_multilingual_uncased_ft | 0.787 | 0.546 | 0.268 | 0.636 | 0.899 |
