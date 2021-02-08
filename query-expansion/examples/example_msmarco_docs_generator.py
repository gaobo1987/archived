import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from bert_embedder import BertEmbedder
from pyserini import collection, index
from tqdm import tqdm
from utils import *
from bert_embedder import BertEmbedder
import math
from tqdm import tqdm
# bert = BertEmbedder('../models/bert/bert-base-uncased')


def doc_generator():
    doc_collection = collection.Collection('TrecCollection', 'collections/msmarco-doc/')
    generator = index.Generator('DefaultLuceneDocumentGenerator')
    for fs in doc_collection:
        for doc in fs:
            parsed = generator.create_document(doc)
            docid = parsed.get('id')
            raw = parsed.get('raw')
            content = parsed.get('contents').replace('\n', ' ').replace('\t', ' ').strip()
            content = clean(content)
            yield docid


def batch_doc_generator(batch_size):
    batch = []
    for docid, content in doc_generator():
        batch.append((docid, content))
        if len(batch) == batch_size:
            result = batch.copy()
            batch = []
            yield result


def dummy_generator():
    for i in range(0, 10000):
        yield i


logfile = open('logs.txt', 'w')


def worker(param):
    s = f'hello {param}\n'
    logfile.write(s)
#     print(s)


if __name__ == '__main__':
    from multiprocessing import Pool
    batch_size = 2
    # gen = batch_doc_generator(batch_size)
    gen = doc_generator()
    # gen = dummy_generator()

    p = Pool(processes=2)
    p.map(worker, gen)
    p.close()
    p.join()

