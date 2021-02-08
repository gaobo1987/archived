import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from bert_embedder import BertEmbedder
from pyserini.search import get_topics
from tqdm import tqdm
import h5py
import numpy as np

CUT_NUM = 10000
DOCS_PATH = f'../data/msmarco-docs-{CUT_NUM}.tsv'
BERT_MODEL_PATH = '../models/bert/bert-base-uncased'
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 120
TOTAL_DOCS = 3213835
DOC_VECS_PATH = f'..data/msmarco-docs-vecs-{MAX_SEQ_LENGTH}.h5'
QUERY_VECS_PATH = f'..data/msmarco-queries-vecs-{MAX_SEQ_LENGTH}.h5'


def doc_generator(docs_path):
    with open(docs_path, 'r') as f:
        for l in f:
            splits = l.split('\t')
            docid = splits[0]
            doc = splits[1][:-1]
            yield docid, doc


def embed_batch_docs_h5(bert, h5file, batch_ids, batch_docs):
    vecs = bert.embed_v1(batch_docs)
    for docid, vec in zip(batch_ids, vecs):
        h5file.create_dataset(docid, data=vec)


def embed_docs_h5():
    h5file = h5py.File(DOC_VECS_PATH, 'w')
    bert = BertEmbedder(BERT_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
    batch_ids = []
    batch_docs = []
    count = 0
    for docid, doc in tqdm(doc_generator(DOCS_PATH), total=TOTAL_DOCS):
        if count > 1691520:
            batch_ids.append(docid)
            batch_docs.append(doc)
            if len(batch_ids) == BATCH_SIZE:
                embed_batch_docs_h5(bert, h5file, batch_ids, batch_docs)
                batch_ids.clear()
                batch_docs.clear()
        count += 1
    if len(batch_ids) > 0:
        print('final round')
        embed_batch_docs_h5(bert, h5file, batch_ids, batch_docs)
    h5file.close()


def embed_docs_npy(k=1):
    print(f'===========================  K={k} ==========================================')
    SUB = f'-sub-{k}'
    DOCS_PATH = f'../data/msmarco-docs-{CUT_NUM}{SUB}.tsv'
    DOC_VECS_PATH = f'../data/msmarco-docs-{CUT_NUM}{SUB}-vecs-{MAX_SEQ_LENGTH}.npy'
    f = open(DOCS_PATH, 'r')
    total = 0
    for _ in f:
        total += 1
    f.close()
    d = {}
    bert = BertEmbedder(BERT_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
    print('Batch size: ', BATCH_SIZE)
    gen = doc_generator(DOCS_PATH)
    batch_ids = []
    batch_docs = []
    for docid, doc in tqdm(gen, total=total):
        batch_ids.append(docid)
        batch_docs.append(doc)
        if len(batch_ids) == BATCH_SIZE:
            batch_vecs = bert.embed_v3(batch_docs)
            for _id, _vec in zip(batch_ids, batch_vecs):
                d[_id] = _vec
            batch_ids.clear()
            batch_docs.clear()

    if len(batch_ids) > 0:
        print('final round')
        batch_vecs = bert.embed_v3(batch_docs)
        for _id, _vec in zip(batch_ids, batch_vecs):
            d[_id] = _vec
        batch_ids.clear()
        batch_docs.clear()

    print(f'saving to {DOC_VECS_PATH}...')
    np.save(DOC_VECS_PATH, d)
    print('Completed sub k=', k)


def embed_queries_h5():
    h5file = h5py.File(QUERY_VECS_PATH, 'w')
    bert = BertEmbedder(BERT_MODEL_PATH)
    topic_set = 'msmarco_doc_dev'
    topics = get_topics(topic_set)
    print(f'{len(topics)} topics total')

    for query_id in tqdm(topics):
        query = topics[query_id]['title']
        print(query_id, query)
        vecs = bert.embed_v1([query])
        h5file.create_dataset(str(query_id), data=vecs[0])

    h5file.close()


if __name__ == '__main__':
    for k in range(3,12):
        embed_docs_npy(k)
    # embed_queries_h5()




