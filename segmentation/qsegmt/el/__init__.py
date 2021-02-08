import os
from qsegmt import Segmenter, do_segmentation
from qsegmt.utils import load_spacy_model

CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
# =================== spaCy model params ===================
SPACY_MODEL_NAME = 'spacy_default'
SPACY_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', SPACY_MODEL_NAME))
# === lemma dictionary based on Universal Dependencies Greek train and dev sets ====
LEMMA_DICT_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', 'el_ud_lemma_dict.tsv'))

LABEL_MAP = {
    'LOC': 'LOC',
    'PERSON': 'PER',
    'PER': 'PER',
    'FAC': 'MISC',
    'NORP': 'MISC',
    'ORG': 'ORG',
    'GPE': 'LOC',
    'PRODUCT': 'MISC',
    'EVENT': 'MISC',
    'WORK_OF_ART': 'MISC',
    'LAW': 'MISC',
    'LANGUAGE': 'MISC',
    'DATE': 'MISC',
    'TIME': 'MISC',
    'PERCENT': 'MISC',
    'MONEY': 'MISC',
    'QUANTITY': 'MISC',
    'ORDINAL': 'MISC',
    'CARDINAL': 'MISC',
    'MISC': 'MISC',
}


def create_spacy_lemma_component():
    lemma_lookups = {}
    r = open(LEMMA_DICT_PATH, 'r', encoding='utf8')
    for l in r:
        parts = l[:-1].split('\t')
        lemma_lookups[parts[0]] = parts[1]
    r.close()

    def lemma_component(doc):
        for t in doc:
            term = t.text
            if term in lemma_lookups:
                t.lemma_ = lemma_lookups[term]
        return doc

    return lemma_component


# Segmentation for Greek
class ELSegmenter(Segmenter):

    def __init__(self):
        # load the spaCy dutch nlp model
        self.model = load_spacy_model(SPACY_MODEL_PATH)
        lemma_component = create_spacy_lemma_component()
        self.model.add_pipe(lemma_component, name="extra_lemma_lookup", last=True)
        self.label_map = LABEL_MAP

    def segment(self, input_string):
        return do_segmentation(self.model, input_string, use_custom_ner=False, label_map=self.label_map)
