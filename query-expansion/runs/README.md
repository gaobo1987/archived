This folder contains runfiles that are the result of running queries
in the msmarco-doc dataset on a pyserini search engine that is indexed
based on the same dataset.

The baseline search scoring is BM25, on top of which we add query expansion 
techniques to see if the output ranked documents are closer to or 
further away from the human judgements.

Explanations on filenames:
- run-msmarco-doc-bm25-nn*:
	derive expanded query terms based on nearest 
	neighbor search in an word embedding, 
	nn3 means, top 3 neighbors, nn5 means top 5 neighbors. 
	v1 means the version 1 implementation of this method.
- run-msmarco-doc-bm25-wordnet-*:
	derive expanded query terms based on synonyms 
	extracted from wordnet, there is no explicit limit 
	on number of neighbors.
- run-msmarco-doc-random-*:
	no query expansion applied, 
	just permutate the original query terms to see 
	if it influences the eventual bm25 ranking, 
	the answer is no, it has no effect.
- run-msmarco-doc-bm25-prf-k*-d*-v*:
	derive expanded query terms based on 
	pseudo relevance feedback, k is the number of expanded terms, 
	d is the number of top documents as relevant documents, 
	v1 means the version 1 implementation of this method.
- run-msmarco-doc-bm25-rm3-k*-d*-qw*:
    derive expanded query terms based on built-in rm3 method,
    k is the number of expanded terms, d is the number of top
    documents to consider, and qw is the initial query weight.
  