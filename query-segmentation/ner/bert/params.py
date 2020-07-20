import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME
)
from os.path import dirname, abspath


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), ())

# demorgen2000
# wikiner
# parliamentary
# meantimenews
# europeananews
# all
DATA_NAME = 'all'
PARENT_DIR = dirname(dirname(abspath(__file__)))
DATA_DIR = os.path.join(PARENT_DIR, 'qsegmt', 'nl', 'data', DATA_NAME)
LABELS_DIR = os.path.join(DATA_DIR, 'labels.txt')
LABELS = get_labels(LABELS_DIR)
OUTPUT_DIR = os.path.join('output')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# allowed model types:
# 'distilbert', 'camembert', 'xlm', 'xlm-roberta', 'roberta', 'bert', 'xlnet', 'albert', 'electra'
# see more examples: https://huggingface.co/transformers/examples.html
# see more models: https://huggingface.co/models

# bert - bert-base-multilingual-cased
# bert - bert-base-dutch-cased
# roberta - robbert-dutch-books
MODEL_TYPE = 'bert'
MODEL_NAME = 'bert-base-dutch-cased'
# MODEL_NAME_OR_PATH = os.path.join('pretrained', MODEL_NAME)
MODEL_NAME_OR_PATH = MODEL_NAME
TOKENIZER_ARGS = {
    'do_lower_case': False,
    'strip_accents': True,
    'keep_accents': False,
    'use_fast': False
}
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index
MAX_SEQ_LENGTH = 128
ADAM_EPSILON = 1e-8
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1
BATCH_SIZE = 32
NUM_EPOCHS = 1


print('*********** Parameters **********')
print('WEIGHTS_NAME: ', WEIGHTS_NAME)
print('DEVICE: ', DEVICE)
print('DATA_DIR: ', DATA_DIR)
print('MODEL_NAME: ', MODEL_NAME)
print('MAX_SEQ_LENGTH: ', MAX_SEQ_LENGTH)
print('ADAM_EPSILON: ', ADAM_EPSILON)
print('LEARNING_RATE: ', LEARNING_RATE)
print('BATCH_SIZE: ', BATCH_SIZE)
print('NUM_EPOCHS: ', NUM_EPOCHS)
print('*********************************')
