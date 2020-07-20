import os
import torch
from torch.nn import CrossEntropyLoss
from qsegmt import logger


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'device: {DEVICE}')
CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
# =================== spaCy model params ===================
SPACY_MODEL_NAME = 'nl_core_news_sm-2.2.5'
SPACY_MODEL_PATH = os.path.join(CUR_DIR, 'models', SPACY_MODEL_NAME)

# =================== bert model params ===================
BERT_MODEL_TYPE = 'bert'
BERT_MODEL_NAME = 'bert-base-dutch-cased-all'
BERT_MODEL_PATH = os.path.join(CUR_DIR, 'models', BERT_MODEL_NAME)
BERT_LABELS_DIR = os.path.join(CUR_DIR, BERT_MODEL_PATH, 'labels.txt')
BERT_LABELS = get_labels(BERT_LABELS_DIR)
BERT_TOKENIZER_ARGS = {
    'do_lower_case': False,
    'strip_accents': True,
    'keep_accents': False,
    'use_fast': False
}
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
