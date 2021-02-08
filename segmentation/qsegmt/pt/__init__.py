import os
import json
from spacy.attrs import ORTH
from qsegmt import Segmenter, do_segmentation
from qsegmt.utils import load_spacy_model, create_spacy_bert_ner_component
from qsegmt.pt.lemmatizer import Lemmatizer


CUR_DIR, _ = os.path.split(__file__)
CUR_DIR = os.path.abspath(CUR_DIR)
# =================== spaCy model params ===================
SPACY_MODEL_NAME = 'pt_core_news_sm-2.2.5'
SPACY_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', SPACY_MODEL_NAME))
# =================== bert model params ===================
BERT_NER_MODEL_NAME = 'bert-base-portuguese-cased-ft'
BERT_NER_MODEL_PATH = os.path.abspath(os.path.join(CUR_DIR, 'models', BERT_NER_MODEL_NAME))
# =================== lemmatizer params ===================
LOOKUPS_PATH = os.path.abspath(os.path.join(SPACY_MODEL_PATH, 'vocab'))
LEMPORT_CONFIG = os.path.join(CUR_DIR, 'LemPyPort', 'resources', 'config', 'lemport.properties')
# =================== tokenizer params ===================
SPECIAL_RULES_PATH = os.path.abspath(os.path.join(CUR_DIR, 'tokenizer_special_rules.json'))


# Segmentation for Portuguese
class PTSegmenter(Segmenter):

    def __init__(self):
        # load the spaCy dutch nlp model
        self.model = load_spacy_model(SPACY_MODEL_PATH, disable=['ner'])
        self.model.tagger.vocab.morphology.lemmatizer = Lemmatizer(LOOKUPS_PATH, LEMPORT_CONFIG)
        ner_component = create_spacy_bert_ner_component(BERT_NER_MODEL_PATH)
        self.model.add_pipe(ner_component, name='bert_ner', last=True)
        self.add_tokenizer_rules()

    def add_tokenizer_rules(self):
        with open(SPECIAL_RULES_PATH, 'r', encoding='utf-8') as f:
            special_rules = json.load(f)
        for word, rule in special_rules.items():
            reconstructed_rule = []
            for item in rule:
                reconstructed_rule.append(
                    {
                        ORTH: item['ORTH'],
                    }
                )
            self.model.tokenizer.add_special_case(word, reconstructed_rule)

    def segment(self, input_string):
        return do_segmentation(self.model, input_string, use_custom_ner=True)
