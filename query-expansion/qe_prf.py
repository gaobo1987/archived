import os
import spacy


class QE_PRF:
    def __init__(self, searcher, num_terms_threshold=5, num_docs_threshold=20):
        '''
        :param num_docs_threshold: number of top retrieved documents
        :param num_terms_threshold: number of top terms as expansion terms
        '''
        print('init QE Pseudo Relevance Feedback')
        print(f'num_terms_threshold={num_terms_threshold}, num_docs_threshold={num_docs_threshold}')
        self.searcher = searcher
        self.num_docs_threshold = num_docs_threshold
        self.num_terms_threshold = num_terms_threshold
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, 'models', 'spacy', 'en_core_web_sm-2.3.0')
        self.model = spacy.load(model_path, disable=['tagger', 'parser', 'ner'])
        # self.model = spacy.load(model_path)
        # print(self.model.pipeline)
        self.index = {}

    def __process_doc(self, doc_id, doc_string):
        doc = self.model(doc_string)
        for token in doc:
            if QE_PRF.verify_token(token):
                lemma = token.lemma_
                if lemma not in self.index:
                    self.index[lemma] = {
                        'doc_ids': {doc_id},
                        'freq': 1
                    }
                else:
                    self.index[lemma]['doc_ids'].add(doc_id)
                    self.index[lemma]['freq'] += 1

    def __process_hits(self, hits):
        self.index = {}
        _hits = hits
        if len(hits) > self.num_docs_threshold:
            _hits = hits[0:self.num_docs_threshold-1]
        for hit in _hits:
            doc_id = hit.docid
            doc_string = hit.raw
            self.__process_doc(doc_id, doc_string)

    def expand_query_version_1(self, query):
        # retrieve initial top docs
        hits = self.searcher.search(query, self.num_docs_threshold)
        # construct mini index for term stats
        self.__process_hits(hits)
        # create expansion terms
        doc = self.model(query)
        remaining_index = {}
        query_terms = set()
        for token in doc:
            if QE_PRF.verify_token(token):
                query_terms.add(token.lemma_)
        for term in self.index:
            if term not in query_terms:
                remaining_index[term] = self.index[term]
        expansion_dict = {k:v for k,v in
                           sorted(remaining_index.items(), key=lambda item: item[1]['freq'], reverse=True)}
        expansion_terms = list(expansion_dict.keys())[0:self.num_terms_threshold]
        return query + ' ' + ' '.join(expansion_terms)

    @staticmethod
    def verify_token(t):
        return not (t.is_stop or t.is_punct or t.like_url or t.is_bracket
                    or t.is_currency or t.is_space or t.is_quote or t.like_num)
