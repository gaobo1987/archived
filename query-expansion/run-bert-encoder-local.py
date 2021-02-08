import os
import torch
print('torch version: ', torch.__version__)
from bert_embedder import BertEmbedder
from tqdm import tqdm
import numpy as np

print('===================================== run  ============================================')
ROOT_DIR = os.path.join('')
MAX_SEQ_LENGTH = 128
EMBED_VERSION = 'v3'
CUT_NUM = 1000 if MAX_SEQ_LENGTH == 128 else 10000
DOCS_PATH = os.path.join(ROOT_DIR, f'data/msmarco-docs-{CUT_NUM}.tsv')
QUERIES_PATH = os.path.join(ROOT_DIR, 'data/msmarco-queries.tsv')
BERT_MODEL_PATH = os.path.join(ROOT_DIR, 'models/bert/bert-base-uncased')
BATCH_SIZE = 480
TOTAL_DOCS = 3213835
TOTAL_QUERIES = 5193
BERT_MODEL = BertEmbedder(BERT_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
EMBED_FUNC = None
if EMBED_VERSION == 'v1':
    EMBED_FUNC = BERT_MODEL.embed_v1
elif EMBED_VERSION == 'v2':
    EMBED_FUNC = BERT_MODEL.embed_v2
elif EMBED_VERSION == 'v3':
    EMBED_FUNC = BERT_MODEL.embed_v3

print('embedding version: ', EMBED_VERSION)
print('docs path: ', DOCS_PATH)
print('max seq length: ', MAX_SEQ_LENGTH)
print('batch size: ', BATCH_SIZE)

# mox.file.copy(source, destination)
# s3://obs-app-2020032316522904031/b00563677/output/msmarco-query-vecs-128.h5
# /home/work/user-job-dir/b00563677/output/msmarco-query-vecs-128.h5
DOC_VECS_PATH = os.path.join(ROOT_DIR, 'output', f'msmarco-docs-{CUT_NUM}-vecs-{MAX_SEQ_LENGTH}-{EMBED_VERSION}.npy')
QUERY_VECS_PATH = os.path.join(ROOT_DIR, 'output', f'msmarco-queries-vecs-{MAX_SEQ_LENGTH}-{EMBED_VERSION}.npy')


def doc_generator(docs_path):
    with open(docs_path, 'r') as f:
        for l in f:
            splits = l.split('\t')
            docid = splits[0]
            doc = splits[1][:-1]
            yield docid, doc


def embed_batch_docs(bert, h5file, batch_ids, batch_docs):
    vecs = bert.embed(batch_docs)
    for docid, vec in zip(batch_ids, vecs):
        h5file.create_dataset(docid, data=vec)


def embed_docs(docs_path):
    print('================================ embed docs =================================')
    d = {}
    batch_ids = []
    batch_docs = []
    count = 0
    for docid, doc in tqdm(doc_generator(docs_path)):
        count += 1
        if count > 30000:
            print(count, docid)
            batch_ids.append(docid)
            batch_docs.append(doc)
            if len(batch_ids) == BATCH_SIZE:
                batch_vecs = EMBED_FUNC(batch_docs)
                for _id, _vec in zip(batch_ids, batch_vecs):
                    d[_id] = _vec
                batch_ids.clear()
                batch_docs.clear()

    if len(batch_ids) > 0:
        print('final round')
        batch_vecs = EMBED_FUNC(batch_docs)
        for _id, _vec in zip(batch_ids, batch_vecs):
            d[_id] = _vec
        batch_ids.clear()
        batch_docs.clear()

    print('================================ save docs vecs =================================')
    print(f'saving to {DOC_VECS_PATH}...')
    np.save(DOC_VECS_PATH, d)


def embed_queries():
    print('================================ embed queries =================================')
    d = {}
    queries = []
    with open(QUERIES_PATH, 'r') as f:
        for l in f:
            splits = l.replace('\n', '').split('\t')
            query_id = splits[0]
            query = splits[1]
            queries.append((query_id, query))

    print(f'{len(queries)} topics total')
    count = 0
    for query_id, query in tqdm(queries, total=len(queries), disable=True):
        print('query', count, float(count) / TOTAL_QUERIES)
        count += 1
        vecs = EMBED_FUNC([query])
        d[query_id] = vecs[0]

    print('================================ save queries vecs =================================')
    print(f'saving to {QUERY_VECS_PATH}...')
    np.save(QUERY_VECS_PATH, d)


if __name__ == '__main__':
    embed_queries()
    for k in range(5,6):
        docs_path = os.path.join('data', f'msmarco-docs-10000-sub-{k}.tsv')
        embed_docs(docs_path)

