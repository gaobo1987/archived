## Benchmarking scores for version 0.6.0

### Polish models highlights

#### Named Entity Recognition

##### Public Datasets

|dataset| f1_strict| precision_strict| recall_strict| f1_exact| precision_exact| recall_exact|
|-------|----------|-----------------|--------------|---------|----------------|-------------|
|poleval| 0.8018| 0.8083| 0.7953| 0.8536| 0.8606| 0.8467|

##### 100 queries Evaluation set
Data to be collected

#### Lemmatization and POS

##### Public Datasets (Three universal dependencies polish datasets)

|dataset name|words| Lemmatization Accuracy| POS f1| POS Precision| POS Recall|
|-----------|------|------|------|------|------|
|pl_lfg_test|350K |94.265%| 83.114%| 83.556%| 86.432%|
|pl_pdb_test|130K |93.102%| 82.619%| 82.732%| 85.936%|
|pl_pud_test|18K |89.641%| 75.996%| 75.747%| 81.233%|

#### 100 queries Evaluation set (All tasks)
Data to be collected

#### Speed Benchmarking For all the functionalities

Hardware: Intel core i7-8550U CPU @ 1.8GHz, 4 cores
Total number of sentences for benchmarking : 56400 Sentence
Average number of tokens per sentence: 9.36
Average time per sentence: 10.19ms



### Dutch models highlights

#### Named Entity Recognition

##### Public Datasets

|dataset| f1_strict| precision_strict| recall_strict| f1_exact| precision_exact| recall_exact|
|-------|----------|-----------------|--------------|---------|----------------|-------------|
|demorgen2000| 0.8715| 0.8680| 0.8750| 0.9474| 0.9436| 0.9512|
|wikiner| 0.9114| 0.9019| 0.9211| 0.9505| 0.9406| 0.9607|
|europeananews| 0.5812| 0.5832| 0.5792| 0.6004| 0.6025| 0.5983|
|meantimenews| 0.3508| 0.3479| 0.3538| 0.3666| 0.3635| 0.3697|
|parliamentary| 0.7309| 0.6875| 0.7801| 0.8073| 0.7594| 0.8617|

##### 100 queries Evaluation set

| f1_strict| precision_strict| recall_strict| f1_exact| precision_exact| recall_exact|
|----------|-----------------|--------------|---------|----------------|-------------|
| 0.65| 0.7359| 0.5821| 0.70| 0.7925| 0.6269|

#### Lemmatization and POS

##### 100 queries Evaluation set

Task| Accuracy | weighted_f1| weighted_precision| weighted_recall|
|----------|---|----------|-----------------|--------------|
|Lemmatization| 0.9208 | 0.9182| 0.9176| 0.9208|
|POS| 0.8258 | 0.8416| 0.8416| 0.8258|

#### Speed Benchmarking For all the functionalities
12ms per query, one query per batch; One thread; One 8-core CPU (3.60GHz), 16G RAM; One Nvidia GTX 1070 GPU, 8G VM.


### Greek models highlights

#### Named Entity Recognition

##### [Spacy NER dataset](https://onebox.huawei.com/#teamspaceFile/1/153/4789876)

| f1_strict| precision_strict| recall_strict| f1_exact| precision_exact| recall_exact|
|----------|-----------------|--------------|---------|----------------|-------------|
| 0.84| 0.77| 0.92| 0.85| 0.79 | 0.94|

#### Lemmatization and POS

##### [Universal Dependencies Greek Dataset (corrected test subset)](https://onebox.huawei.com/p/fd724970eb76780ce63a1d696770701d)

Task| Accuracy | weighted_f1| weighted_precision| weighted_recall|
|----------|---|----------|-----------------|--------------|
|POS | 0.901 | 0.916 | 0.941 | 0.901 |
|Lemma | 0.901 | 0.913 | 0.938 | 0.901 | 

#### Speed Benchmarking For all the functionalities
2.7ms per query, one query per batch; One thread; One 8-core CPU (3.60GHz), 16G RAM; One Nvidia GTX 1070 GPU, 8G VM.