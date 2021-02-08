from os.path import join
import argparse

# ========================= Default parameters ===========================
# Path params
# Retriever - ElasticSearch
ES_HOST = 'localhost'
ES_PORT = 9200
# 'wikipedia': original document-level index
# 'wikipedia_en': alternative name to the document-level index
# 'wikipedia_paragraph': paragraph-level index
# 'wikipedia_50': fixed-length 50 tokens index (no stride)
# 'wikipedia_100': fixed-length 100 tokens index (no stride)
# 'wikipedia_200': fixed-length 200 tokens index (no stride)
# 'wikipedia_100_stride_50': fixed-length 100 tokens index with stride 50
ES_INDEX_NAME = 'wikipedia'

# Retriever - DPR
DPR_MODEL_PATH = '../models/dpr/multi_hf_bert_base.cp'
# Datasets
DATA_DIR = join('..', 'data')
DATASETS = [
    join(DATA_DIR, 'squad2', 'squad-1000-formatted.json'),
    join(DATA_DIR, 'naturalQuestions', 'nq-1000-formatted.json'),
    # join(DATA_DIR, 'squad2', 'squad2-dev.json'),
    # join(DATA_DIR, 'naturalQuestions', 'naturalQuestions-dev-clean.json'),
    # join(DATA_DIR, 'quasarT', 'quasarT-dev.json'),
    # join(DATA_DIR, 'searchQA', 'searchQA-dev.json'),
    # join(DATA_DIR, 'triviaQA', 'triviaQA-dev.json'),
    # join(DATA_DIR, 'wikiQA', 'wikiQA-dev.json')
]
# Readers
MODELS_DIR = join('..', 'models')
READERS = [
    join(MODELS_DIR, 'electra-base-squad2')
]

# Model params
USE_GPU = True
RETRIEVER_ES_TOP_K = 100
RETRIEVER_DPR_TOP_K = 20
READER_TOP_K = 20
FAISS_INDEX_DIMENSION = 768
SEED = 42
SUBSET = 1000

# ========================= Args ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--index', default=ES_INDEX_NAME, type=str, required=False)
parser.add_argument('--host', default=ES_HOST, type=str, required=False)
parser.add_argument('--port', default=ES_PORT, type=str, required=False)
parser.add_argument('--use_gpu', default=USE_GPU, type=bool, required=False)
parser.add_argument('--retriever_es_k', default=RETRIEVER_ES_TOP_K, type=int, required=False)
parser.add_argument('--retriever_dpr_k', default=RETRIEVER_DPR_TOP_K, type=int, required=False)
parser.add_argument('--reader_k', default=READER_TOP_K, type=int, required=False)
parser.add_argument('--faiss_dimension', default=FAISS_INDEX_DIMENSION, type=int, required=False)
parser.add_argument('--seed', default=SEED, type=int, required=False)
parser.add_argument('--subset', default=SUBSET, type=int, required=False)
args = parser.parse_args()
