### Index Wikipedia (en) in ElasticSearch

##### Document-level Indexing
```shell script
python index.py /path/to/wikipedia_dump_extracted/
```
INFO: 
* total of page: 10,875,238, 
* total of article page: 6,144,574; 
* total of used article page: 6,005,733

##### Paragraph-level Indexing
```shell script
python index.py /path/to/wikipedia_dump_extracted/ --paragraph
```
INFO: 
* total of page: 10,875,238, 
* total of article page: 6,144,574; 
* total of used article page: 6,005,733
