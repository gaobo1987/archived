from pyserini import analysis, index

index_reader = index.IndexReader('../indexes/msmarco-doc/')

# Get the total number of documents in a collection
# For msmarco-doc collection, D_N = 3213835
# from tqdm import tqdm
#
# D_N = 0
from pyserini import collection, index
#
# collection = collection.Collection('TrecCollection', '../collections/msmarco-doc/')
#
# for (i, fs) in enumerate(collection):
#     for doc in tqdm(fs):
#         D_N += 1

# D_N = 3213835 # total number of documents in a collection
# D_R = 20 # the number of documents selected as relevant by the user,
#         # in case of PRF, this number can be the number of top ranked documents, 20 is usually a good choice
# n = 0   # the number of documents containing the term
# r = 0   # the number of (pseudo-)relevant documents containing the term


import itertools
n = 10000000
for term in itertools.islice(index_reader.terms(), n, n+10):
    print(f'{term.term} (df={term.df}, cf={term.cf})')
