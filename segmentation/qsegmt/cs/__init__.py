import os
from qsegmt import Segmenter, do_segmentation
from qsegmt.utils import load_spacy_stanza_model, create_spacy_bert_ner_component

# =================== model params ===================
CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
STANZA_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', 'stanza_resources'))
BERT_NER_MODEL_NAME = 'bert_base_multilingual_uncased_cs_ft'
BERT_NER_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', BERT_NER_MODEL_NAME))


# Segmentation for Czech
class CSSegmenter(Segmenter):

    def __init__(self):
        self.config = {
            'lang': 'cs',
            'processors': 'tokenize,mwt,pos,lemma',
            'dir': STANZA_MODEL_PATH,
            'use_gpu': True,
            'logging_level': 'ERROR'
        }
        self.model = load_spacy_stanza_model(self.config)
        ner_component = \
            create_spacy_bert_ner_component(BERT_NER_MODEL_PATH)
        self.model.add_pipe(ner_component, name="bert_ner", last=True)

    def segment(self, input_string):
        return do_segmentation(self.model, input_string, use_custom_ner=True)
