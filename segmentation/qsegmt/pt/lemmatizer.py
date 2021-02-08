"""Portuguese Lemmatizer that uses some PyPort lemmatizer for some POS and SpaCy lookup for the rest
"""
import os
from spacy.parts_of_speech import NAMES as UPOS_NAMES
from spacy.lookups import Lookups
from qsegmt.pt.LemPyPort.LemFunctions import *

LEMPORT_TAGS = ('ADV', 'NOUN', 'PRON', 'VERB', 'ADJ')


class Lemmatizer(object):
    def __init__(self, lookups_path, lemport_config):
        loaded_data = nlpyport_lematizer_loader(lemport_config)
        self.pos_lemmatizers = loaded_data[:8]
        self.ranking = loaded_data[8]
        self.novo_dict = loaded_data[9]
        self.lookups = Lookups()
        self.lookups.from_disk(lookups_path)

    @staticmethod
    def normalize_tag(univ_pos):
        if univ_pos == 'NOUN':
            return 'n'
        if univ_pos == 'VERB':
            return 'v'
        else:
            return univ_pos

    def __call__(self, token, univ_pos, morphology=None):
        if isinstance(univ_pos, int):
            univ_pos = UPOS_NAMES.get(univ_pos, 'X')
        if univ_pos in LEMPORT_TAGS:
            univ_pos = Lemmatizer.normalize_tag(univ_pos)
            args = self.pos_lemmatizers + (token.lower(), univ_pos.lower(), self.ranking, self.novo_dict)
            lemma = all_normalizations(*args)
            return [lemma]
        else:
            lookup_table = self.lookups.get_table('lemma_lookup')
            lemma = lookup_table.get(token, token)
            return [lemma]
