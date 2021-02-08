
import os
import moxing as mox
mox.file.shift('os', 'mox')

# set jvm, it seems that oneAI instance does not have jvm
# print('set java home')
# os.environ["JAVA_HOME"] = '/usr/lib/jvm/default-java'
# os.system('pip install pyserini')
# from pyserini.search import get_topics

import torch
print('torch version: ', torch.__version__)

print('install transformers...')
# transformers_dir = os.path.join(os.getcwd(), 'src', 'transformers')
# os.system(f'pip install {transformers_dir}')
os.system('pip install transformers')
os.system('pip install h5py')



from bert_embedder import BertEmbedder
from tqdm import tqdm
import numpy as np

print('===================================== run  ============================================')
OBS_ROOT = os.path.join('s3://obs-app-2020032316522904031/b00563677/')
ROOT_DIR = os.path.join('/home/work/user-job-dir/b00563677/')
MAX_SEQ_LENGTH = 512
EMBED_VERSION = 'v1'
CUT_NUM = 1000 if MAX_SEQ_LENGTH == 128 else 10000
DOCS_PATH = os.path.join(ROOT_DIR, f'data/msmarco-docs-{CUT_NUM}.tsv')
QUERIES_PATH = os.path.join(ROOT_DIR, 'data/msmarco-queries.tsv')
BERT_MODEL_PATH = os.path.join(ROOT_DIR, 'models/bert-base-uncased')
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
DOC_VECS_PATH_OBS = os.path.join(OBS_ROOT, 'output', f'msmarco-docs-{CUT_NUM}-vecs-{MAX_SEQ_LENGTH}-{EMBED_VERSION}.npy')
QUERY_VECS_PATH_OBS = os.path.join(OBS_ROOT, 'output', f'msmarco-queries-vecs-{MAX_SEQ_LENGTH}-{EMBED_VERSION}.npy')


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


def embed_docs():
    print('================================ embed docs =================================')
    d = {}
    count = 0
    try:
        batch_ids = []
        batch_docs = []
        for docid, doc in tqdm(doc_generator(DOCS_PATH), total=TOTAL_DOCS, disable=True):
            # if count % 1000 == 0:
            print('-->', count, float(count) / TOTAL_DOCS, docid)
            count += 1
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
    except Exception as e:
        print('Exception count: ', count)
        print('Exception: ', str(e))

    print('================================ save docs vecs =================================')
    print(f'saving to {DOC_VECS_PATH}...')
    np.save(DOC_VECS_PATH, d)
    print('================================ move docs vecs =================================')
    print(f'moving to {DOC_VECS_PATH_OBS}...')
    mox.file.copy(DOC_VECS_PATH, DOC_VECS_PATH_OBS)


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
    print('================================ move queries vecs =================================')
    print(f'moving to {QUERY_VECS_PATH_OBS}...')
    mox.file.copy(QUERY_VECS_PATH, QUERY_VECS_PATH_OBS)


if __name__ == '__main__':
    embed_queries()
    embed_docs()

