import os
from qsegmt import Segmenter
from qsegmt.utils import load_spacy_model, do_segmentation


CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
# =================== spaCy model params ===================
SPACY_MODEL_NAME = 'spacy_default'
SPACY_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', SPACY_MODEL_NAME))
# =================== bert model params ===================
BERT_MODEL_TYPE = 'bert'
BERT_MODEL_NAME = 'bert_base_greek_uncased_ft'
BERT_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', BERT_MODEL_NAME))


# Segmentation for Greek
class ELSegmenter(Segmenter):

    def __init__(self):
        # load the spaCy dutch nlp model
        self.model = load_spacy_model(SPACY_MODEL_PATH)

    def segment(self, input_string):
        return do_segmentation(self.model, input_string, use_custom_ner=False)
