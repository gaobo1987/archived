# coding: utf-8
import os
import gzip
import shutil
from spacy.symbols import POS, ADJ, ADP, ADV, AUX, CCONJ, INTJ, NOUN, NUM, PART, PRON, SCONJ, VERB, X, DET, PROPN, PUNCT, SYM
from os import path
from spacy.lookups import Lookups

currdir=path.dirname(__file__)
compressed_lookup_data = os.path.join(currdir, 'lookups.bin.gz')
decompressed_lookup_data = os.path.join(currdir, 'lookups.bin')
if os.path.isfile(compressed_lookup_data):
    with gzip.open(compressed_lookup_data, 'rb') as f_in:
        with open(decompressed_lookup_data, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


lkp_tables={}


##def bin_lemma_search(a, x):
##    # binary search to speed-up lookup
##    i = bisect_left(a, (x,''))
##    if i != len(a) and a[i][0] == x:
##        return a[i][1]
##        # returns the second part of the pair, i.e. the lemma
##    return None
##
class PolishLemmatizer(object):
    # This lemmatizer implements lookup lemmatization based on
    # the Morfeusz dictionary (morfeusz.sgjp.pl/en) by Institute of Computer Science PAS
    # It utilizes some prefix based improvements for
    # verb and adjectives lemmatization, as well as case-sensitive
    # lemmatization for nouns
    @classmethod
    def load(cls, path, index=None, exc=None, rules=None, lookup=None):
        return cls(index, exc, rules, lookup)

    def __init__(self, index=None, exceptions=None, rules=None, lookup=None):
        # this lemmatizer is lookup based, so it does not require an index, exceptionlist, or rules
        # the lookup tables are imported from JSON files 
        self.INT_TO_POS={POS:'POS',
            ADJ:'ADJ',
            ADP:'ADP',
            ADV:'ADV',
            AUX:'AUX',
            CCONJ:'CCONJ',
            DET:'DET',
            INTJ:'INTJ',
            NOUN:'NOUN',
            NUM:'NUM',
            PART:'PART',
            PRON:'PRON',
            PROPN:'PROPN',
            PUNCT:'PUNCT',
            SCONJ:'SCONJ',
            VERB:'VERB',
            X:'X'
          }
        self.lookups = Lookups()
        self.lookups.from_disk(currdir)
        self.lemma_lookups = {}
        for tag in ['ADJ', 'ADP', 'ADV', 'AUX', 'NOUN', 'NUM', 'PART', 'PRON', 'VERB', 'X']:
            self.lemma_lookups[tag] = self.lookups.get_table(tag)
        empty_table = self.lookups.add_table('empty', {})
        additional_tags={
                # additional tags outside of the tagmaps range
                 'CCONJ': empty_table,
                 'INTJ': empty_table,
                 'SCONJ': empty_table,
                 'DET' : self.lookups.get_table('X'),
                 'PROPN' : self.lookups.get_table('NOUN'),
                 'PUNCT' : empty_table,
                 'SYM' : empty_table
                 }
        self.lemma_lookups.update(additional_tags)

        
        
    def lemmatize_adj(self, string, morphology):
        # this method utilizes different procedures for adjectives
        # with 'nie' and 'naj' prefixes
        lemmas=[]
        lemma_dict=self.lemma_lookups['ADJ']
        if string[:3]=='nie':
            search_string=string[3:]

            if search_string[:3]=='naj':
                naj_search_string=search_string[3:]
                try:
                    lemma=lemma_dict[naj_search_string]
                    lemmas.append(lemma)
                    return lemmas
                except KeyError:
                    pass

            try:
                lemma=lemma_dict[search_string]
                lemmas.append(lemma)
                return lemmas
            except KeyError:
                pass
            
        if string[:3]=='naj':
            naj_search_string=string[3:]
            try:
                lemma=lemma_dict[naj_search_string]
                lemmas.append(lemma)
                return lemmas
            except KeyError:
                pass
        try:
            lemma=lemma_dict[string]
            lemmas.append(lemma)
            return lemmas
        except KeyError:
            lemmas = [string]
            return lemmas
            
    def lemmatize_verb(self, string, morphology):
        # this method utilizes a differen procedures for verbs
        # with 'nie' prefix
        lemmas=[]
        lemma_dict=self.lemma_lookups['VERB']
        
        if string[:3]=='nie':
            search_string=string[3:]
            try:
                lemma=lemma_dict[search_string]
                lemmas.append(lemma)
                return lemmas
            except KeyError:
                pass
        try:    
            lemma=lemma_dict[string]
            lemmas.append(lemma)
            return lemmas
        except KeyError:
            lemmas = [string]
            return lemmas
        
    def lemmatize_noun(self, string, morphology):
        # this method is case-sensitive, in order to work
        # for incorrectly tagged proper names
        lemmas=[]
        lemma_dict=self.lemma_lookups['NOUN']
        if string!=string.lower():
            try:
                lemma=lemma_dict[string.lower()]
                lemmas.append(lemma)
                return lemmas
            except KeyError:
                try:
                    lemma=lemma_dict[string]
                    lemmas.append(lemma)
                except KeyError:
                    lemmas.append(string.lower())
                return lemmas
        else:
            try:
                lemma=lemma_dict[string]
            except KeyError:
                lemma=string
            lemmas.append(lemma)
            return lemmas
        
    def __call__(self, string, univ_pos, morphology=None):
        if type(univ_pos) == int:
            try:
                univ_pos = self.INT_TO_POS[univ_pos]
            except KeyError:
                univ_pos = self.INT_TO_POS[X]
                
        univ_pos = univ_pos.upper()
        
        if univ_pos == 'NOUN':
            return self.lemmatize_noun(string, morphology)
        
        if univ_pos != 'PROPN':
            string=string.lower()
            
        if univ_pos == 'ADJ':
            return self.lemmatize_adj(string, morphology)
        if univ_pos == 'VERB':
            return self.lemmatize_verb(string, morphology)

        lemmas = []
        lemma_dict = self.lemma_lookups[univ_pos]
        try:
            lemma=lemma_dict[string]
        except KeyError:
            lemma=string.lower()
        lemmas.append(lemma)
        return lemmas

    def lookup(self, string):
        return string.lower()


def lemmatize(string, index, exceptions, trie):
    print('This message should not appear, this lemmatizer should not use the function "lemmatize"')