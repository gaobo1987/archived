### Comparing BERT models trained on different datasets

###### Data description:
All the datasets refer to the "test" sets.

|       Data 	| #sentences 	| #tokens/sent. | 
|--------------:|--------------:|--------------:|
| demorgen2000 	|        5,195 	|        13.3 	|
| wikiner 	    |        27,648 |        18.8 	|
| parliamentary |        502 	|        20.9 	|
| meantimenews  |        534 	|        22.0 	|
| europeananews |        602 	|        100.7 	|

###### F1 scores:
The scores come from the evaluation notebook. 
Note that the F1 score is in strict mode.

|                                     F1 | demorgen2000 | wikiner | parliamentary | meantimenews | europeananews |
|---------------------------------------:|-------------:|--------:|---------------|--------------|---------------|
|                                 stanza |         .886 |    .795 |          .581 |         .213 |          .056 |
|                                  spacy |         .490 |    .527 |          .380 |         .071 |          .039 |
|  dutch-base-dutch-cased (demorgen2000) |     __.904__ |    .794 |          .667 |         .243 |          .363 |
|       dutch-base-dutch-cased (wikiner) |         .725 |__.913__ |          .582 |         .254 |          .336 |
| dutch-base-dutch-cased (parliamentary) |         .080 |    .051 |          .401 |         .020 |          .053 |
| dutch-base-dutch-cased (meantimenews)  |         .030 |    .014 |             0 |         .280 |             0 |
| dutch-base-dutch-cased (europeananews) |         .080 |    .199 |          .116 |         .011 |          .456 |
|           dutch-base-dutch-cased (all) |         .874 |    .903 |      __.700__ |     __.300__ |      __.535__ |

