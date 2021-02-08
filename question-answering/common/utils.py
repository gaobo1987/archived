import json
import faiss
import logging
import numpy as np
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers


def load_json(input_filename):
    with open(input_filename, 'r', encoding='utf8') as f:
        return json.load(f)


def save_json(obj_to_save, output_filename):
    with open(output_filename, 'w', encoding='utf8') as out:
        json.dump(obj_to_save, out, ensure_ascii=False)


def get_faiss_ip_index(d=768, use_gpu=True):
    # build a Inner Product (CPU) index
    index_cpu = faiss.IndexFlatIP(d)
    if use_gpu:
        # claim single GPU resource
        resource = faiss.StandardGpuResources()
        # make it into a gpu index
        index_gpu = faiss.index_cpu_to_gpu(resource, 0, index_cpu)
        return index_gpu
    return index_cpu


def get_elastic_search_document_store(es_host='localhost',
                                      es_port=9200,
                                      es_index_name='wikipedia',
                                      search_fields=['text']):
    return ElasticsearchDocumentStore(host=es_host, port=es_port,
                                      username="", password="",
                                      index=es_index_name,
                                      search_fields=search_fields)


def get_elastic_search_retriever(document_store):
    return ElasticsearchRetriever(document_store=document_store)


def get_dense_passage_retriever(document_store,
                                dpr_model_path,
                                use_gpu=True,
                                batch_size=16,
                                do_lower_case=True):
    return DensePassageRetriever(document_store=document_store,
                                 embedding_model=dpr_model_path,
                                 use_gpu=use_gpu,
                                 batch_size=batch_size,
                                 do_lower_case=do_lower_case)


def get_neural_reader(reader_path, use_gpu=True):
    use_gpu = 0 if use_gpu else -1
    return TransformersReader(model=reader_path, tokenizer=reader_path, use_gpu=use_gpu)


def get_logger(logger_name, log_file_path):
    # list all logger names
    # for name in logging.root.manager.loggerDict:
    #     print(name)
    logging.getLogger('haystack.reader.transformers_utils').setLevel(logging.ERROR)
    logging.getLogger('haystack.reader').setLevel(logging.ERROR)
    logging.getLogger('haystack.retriever.sparse').setLevel(logging.ERROR)
    logging.getLogger('haystack.retriever.dense').setLevel(logging.ERROR)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)
    # create logger with 'spam_application'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_qa_data(data_path, seed=None, subset=None):
    with open(data_path, 'r', encoding='utf8') as f:
        loaded_data = json.load(f)
        if seed is not None and subset is not None:
            np.random.seed(seed=seed)
            subset = min(len(loaded_data), subset)
            subset_indices = np.random.choice(range(len(loaded_data)), subset, replace=False)
            loaded_data = [loaded_data[i] for i in subset_indices]
        return loaded_data


def normalize_min_max(min_val, max_val, val):
    if max_val > min_val:
        return (val - min_val) / (max_val - min_val)
    elif max_val == min_val:
        return 1
    return None
