import os
import random
import spacy


class QE_random:

    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, 'models', 'spacy', 'en_core_web_sm-2.3.0')
        self.model = spacy.load(model_path, disable=['tagger', 'parser', 'ner'])

    def shuffle_query(self, query):
        doc = self.model(query)
        tokens = [t.text for t in doc]
        random.shuffle(tokens)
        shuffled_query = ' '.join(tokens)
        return shuffled_query
