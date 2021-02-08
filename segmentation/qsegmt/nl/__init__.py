import os
from qsegmt import Segmenter, do_segmentation
from qsegmt.utils import load_spacy_model, create_spacy_bert_ner_component

CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
# =================== spaCy model params ===================
SPACY_MODEL_NAME = 'spacy_default'
SPACY_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', SPACY_MODEL_NAME))
# =================== bert model params ===================
BERT_NER_MODEL_NAME = 'bert_base_dutch_cased_ft'
BERT_NER_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', BERT_NER_MODEL_NAME))


# Segmentation for Dutch
class NLSegmenter(Segmenter):

    def __init__(self):
        # load the spaCy dutch nlp model
        self.model = load_spacy_model(SPACY_MODEL_PATH, disable=['ner'])
        ner_component = create_spacy_bert_ner_component(BERT_NER_MODEL_PATH)
        self.model.add_pipe(ner_component, name="bert_ner", last=True)

    def segment(self, input_string):
        return do_segmentation(self.model, input_string, use_custom_ner=True)
