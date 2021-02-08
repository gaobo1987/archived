# query-expansion

### Install requirements

If you have a CPU machine, do 
```shell script
pip install -r requirements-cpu.txt
```

If you have a GPU machine, do
```shell script
pip install -r requirements-gpu.txt
```
Remark: adjust the torch version and cuda version accordingly,
The default is torch-1.5 and cuda-10.1.

### Covert FastText word vectors to spaCy model

```shell script
python -m spacy init-model en models/wiki_en_align --vectors-loc wiki.en.align.vec
```

```python
import spacy
nlp = spacy.load('models/wiki_en_align')
doc = nlp('hello world')
```

### Background materials
* [MUSE](https://github.com/facebookresearch/MUSE)
* [FastText aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html)
* [FAISS](https://github.com/facebookresearch/faiss)
