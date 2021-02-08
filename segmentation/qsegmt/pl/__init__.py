import os
from qsegmt import Segmenter, do_segmentation
from qsegmt.utils import load_spacy_model
from qsegmt.pl.Lemmatizer.lemmatizer import PolishLemmatizer

CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
SPACY_MODEL_NAME = 'pl_model-0.1.0'
SPACY_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', SPACY_MODEL_NAME))

LABEL_MAP = {
    'date': 'MISC',
    'geogName': 'LOC',
    'orgName': 'ORG',
    'persName': 'PER',
    'placeName': 'LOC',
    'time': 'MISC'
}


class PLSegmenter(Segmenter):

    def __init__(self):
        self.model = load_spacy_model(SPACY_MODEL_PATH)
        self.model.tagger.vocab.morphology.lemmatizer = PolishLemmatizer()
        self.label_map = LABEL_MAP

    def segment(self, input_string):
        return do_segmentation(self.model, input_string, use_custom_ner=False, label_map=self.label_map)
