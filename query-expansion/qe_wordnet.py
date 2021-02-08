import spacy
import nltk
from nltk.stem import PorterStemmer
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import os,inspect


class QE_wordnet:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        print('Current directory: ', current_dir)
        nltk_path = os.path.abspath(os.path.join(current_dir, 'models', 'nltk'))
        spacy_en_path = os.path.abspath(os.path.join(current_dir, 'models', 'spacy', 'en_core_web_sm-2.3.0'))
        nltk.data.path.append(nltk_path)
        self.nlp = spacy.load(spacy_en_path)
        self.nlp.add_pipe(WordnetAnnotator(self.nlp.lang), after='tagger')
        self.ps = PorterStemmer()

    def expand_query_version_1(self, query):
        doc = self.nlp(query)
        new_terms = []
        for t in doc:
            synsets = t._.wordnet.synsets()
            if not t.is_stop and not t.is_punct:
                if not synsets:
                    new_terms.append(t.text)
                else:
                    lemmas_for_synset = [lemma for s in synsets for lemma in s.lemma_names()]
                    lemmas_for_synset = [l.replace('_', ' ').replace('-', ' ') for l in lemmas_for_synset]
                    # lemmas_for_synset = [self.ps.stem(l) for l in lemmas_for_synset]
                    new_terms += lemmas_for_synset
        new_terms = list(set(new_terms))
        new_query = ' '.join(new_terms).strip()
        return new_query

    def expand_query_version_2(self, query):
        doc = self.nlp(query)
        new_terms = []
        for t in doc:
            synsets = t._.wordnet.synsets()
            if not t.is_stop and not t.is_punct:
                if not synsets:
                    new_terms.append(t.text)
                else:
                    lemmas_for_synset = [lemma for s in synsets for lemma in s.lemma_names()]
                    lemmas_for_synset = [l.replace('_', ' ').replace('-', ' ') for l in lemmas_for_synset]
                    lemmas_for_synset = [self.ps.stem(l) for l in lemmas_for_synset]
                    new_terms += lemmas_for_synset
        new_terms = list(set(new_terms))
        new_query = ' '.join(new_terms).strip()
        return new_query

