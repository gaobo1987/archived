"""
This script has the sole purpose of taking all the documents from ElasticSearch
and encode them with DPR encoder into vectors, then store these vectors into a file.
"""

import os, sys
import numpy as np
import elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from params import *
from evaluation import *


ES_INDEX_NAME = 'wikipedia_paragraph'
BATCH_COUNT = 184


class BatchWriter:
    def __init__(self, batch_size=1000, output_dir='', output_file_prefix='batch_'):
        self.items = []
        self.count = BATCH_COUNT+1
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.output_file_prefix = output_file_prefix

    def save_batch(self):
        print(f'save batch {self.count}')
        if self.items:
            output_file_path = f'{self.output_file_prefix}{str(self.count)}.npy'
            output_file_path = os.path.join(self.output_dir, output_file_path)
            np.save(output_file_path, self.items)
            self.items = []
            self.count += 1

    def add_item(self, item):
        self.items.append(item)
        if len(self.items) >= self.batch_size:
            self.save_batch()


def get_source_doc_by_field(es, field_pair):
    query_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            field_pair[0]: field_pair[1]
                        }
                    }
                ]
            }
        }
    }
    res = es.search(index=ES_INDEX_NAME, body=query_body)
    assert len(res['hits']['hits']) == 1
    doc = res['hits']['hits'][0]['_source']
    return doc


def get_doc_by_id(es, _id):
    doc = es.get(index=ES_INDEX_NAME, id=_id)
    return doc


def embed_batch(batch_writer, encoder, doc_ids, docs_original, docs_normalized):
    assert len(doc_ids) == len(docs_original)
    assert len(doc_ids) == len(docs_normalized)
    if doc_ids:
        vecs = encoder.embed_passages(docs_normalized)
        for i, vec in enumerate(vecs):
            item = {
                'id': doc_ids[i],
                'text': docs_original[i],
                'vec': vec
            }
            batch_writer.add_item(item)
        del vecs
        doc_ids.clear()
        docs_original.clear()
        docs_normalized.clear()


def main():
    logger = get_logger('run_DPR_Encoder', 'run_DPR_Encoder.log')
    count = 0
    embed_batch_size = 128
    writer_batch_size = 100000
    total_docs = 36323970 # 6005733 document-level docs, 36323970 paragraph-level docs

    try:
        doc_ids = []
        docs_original = []
        docs_normalized = []
        batch_writer = BatchWriter(writer_batch_size)
        encoder = get_dense_passage_retriever(document_store=None,
                                              dpr_model_path=DPR_MODEL_PATH,
                                              use_gpu=USE_GPU,
                                              batch_size=embed_batch_size,
                                              do_lower_case=True)
        es = elasticsearch.Elasticsearch(f'http://{ES_HOST}:{ES_PORT}')
        es_response = scan(
            es,
            index=ES_INDEX_NAME,
            query={"query": {"match_all": {}}}
        )
        for item in tqdm(es_response, total=total_docs):
            count += 1
            if count >= BATCH_COUNT*writer_batch_size:
                _id = item['_id']
                doc_ids.append(_id)
                text = item['_source']['text']
                docs_original.append(text)
                n_text = normalize_text(text)
                docs_normalized.append(n_text)
                if len(docs_normalized) == embed_batch_size:
                    embed_batch(batch_writer, encoder, doc_ids, docs_original, docs_normalized)

        # encode remaining docs
        if doc_ids:
            embed_batch(batch_writer, encoder, doc_ids, docs_original, docs_normalized)

        # save remaining docs
        batch_writer.save_batch()
    except (Exception, KeyboardInterrupt) as e:
        logger.info(e)
        logger.info(f'An error occurred at Count {count}...')


if __name__ == '__main__':
    main()

    # v0 = np.load('batch_0.npy', allow_pickle=True)
    # print(len(v0))
    # print(v0[0]['id'])
    # print(v0[0]['text'])
    # v1 = np.load('vecs_1.npy')
    # print(len(v1))
    # v2 = np.load('vecs_2.npy')
    # print(len(v2))
    # v4 = np.load('batch_4.npy', allow_pickle=True)
    # print(len(v4))
    # print(len(v4[0]['vec']))
