import faiss
import spacy
import numpy as np
import re


class QE_NN:
    def __init__(self, d=300, num_terms_threshold=5):
        self.num_terms_threshold = num_terms_threshold
        model_path = 'models/spacy/wiki_en_align'
        self.model = spacy.load(model_path)
        self.vocab = self.model.vocab
        self.ids = []
        x = []
        for _id in self.vocab.vectors:
            vec = self.vocab.vectors[_id]
            self.ids.append(_id)
            x.append(vec)
        x = np.array(x).astype('float32')
        self.index = faiss.IndexFlatL2(d)  # build the index
        self.index.add(x)

    def _lookup(self, word) -> list or None:
        _id = self.vocab.strings[str(word)]
        if _id in self.vocab.vectors:
            _vec = self.vocab.vectors[_id]
            return _vec
        else:
            # print(f'could not find vector for string "{word}"')
            return None

    def find(self, term, k):
        """
        term: the actual target string term
        q: the numpy double array of query vectors
        k: find the k nearest neighbors
        """
        _vec = self._lookup(term)
        result = []
        if _vec is not None:
            q = np.array([self._lookup(term)])
            D, I = self.index.search(q, k+1)
            neighbor_inds = I[0]
            for i, ind in enumerate(neighbor_inds):
                if i > 0:
                    _id = self.ids[ind]
                    text = self.vocab.strings[_id]
                    text = re.sub(r"[^a-zA-Z0-9]+", ' ', text).strip()
                    # vec = self.vocab.vectors[_id]
                    # print(ind, text, vec[:5])
                    if not text in result:
                        result.append(text)
        return result

    def expand_query_version_1(self, query):
        k = self.num_terms_threshold
        lquery = query.strip().lower()
        doc = self.model(lquery)
        expanded_query = ''
        for t in doc:
            if not t.is_stop and not t.is_punct:
                expanded_query += t.text + ' '
                neighbors = self.find(t.text, 2*k)
                for i,n in enumerate(neighbors):
                    if i < k:
                        expanded_query += n + ' '
        return expanded_query.rstrip()


