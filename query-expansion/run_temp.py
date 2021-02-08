import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"
from tqdm import tqdm
from pyserini.search import SimpleSearcher


def load_qe(mode, num_terms_threshold=5, num_docs_threshold=20, original_query_weight=0.5, searcher=None):
    qe = None
    # init query expansion if needed
    if mode in ['nn', 'wordnet', 'prf']:
        print('====================== verify query expansion ==============================')
        if mode == 'nn':
            from qe_nn import QE_NN
            qe = QE_NN(num_terms_threshold=num_terms_threshold)
        elif mode == 'wordnet':
            from qe_wordnet import QE_wordnet
            qe = QE_wordnet()
        elif mode == 'prf':
            from qe_prf import QE_PRF
            qe = QE_PRF(searcher, num_terms_threshold=num_terms_threshold, num_docs_threshold=num_docs_threshold)
        elif mode == 'rm3':
            # rm3 is built-in, no need to create qe
            print('before setting rm3, is it in use? ', searcher.is_using_rm3())
            searcher.set_rm3(fb_terms=num_terms_threshold, fb_docs=num_docs_threshold,
                             original_query_weight=float(original_query_weight), rm3_output_query=True)
            print('after setting rm3, is it in use? ', searcher.is_using_rm3())
        if qe is not None:
            result = qe.expand_query_version_1('apple')
            print(result)
    return qe


def load_index():
    # sample_query = 'does solar radiation cause global warming'
    # sample_query = 'An apple does not fall far from the tree.'
    sample_query = 'orange'
    print('sample query: ', sample_query)
    print('====================== verify index loading ==============================')
    searcher = SimpleSearcher('indexes/msmarco-doc')
    hits = searcher.search(sample_query)
    for i in range(0, 10):
        print(f'{i + 1:2} {hits[i].docid} {hits[i].score:.5f} {hits[i].raw[:70]}...')
    print('====================== loading queries from topics ==============================')
    from pyserini.search import get_topics
    topic_set = 'msmarco_doc_dev'
    topics = get_topics(topic_set)
    print(f'{len(topics)} topics total')
    return searcher, topics


def run_all_queries(run_filename, topics, searcher, qe):
    with open(run_filename, 'w') as runfile:
        cnt = 0
        print('Running {} queries in total'.format(len(topics)))
        for id in tqdm(topics):
            query = topics[id]['title']
            if qe is not None:
                query = qe.expand_query_version_1(query)
            hits = searcher.search(query, 1000)
            for i in range(0, len(hits)):
                _ = runfile.write('{} Q0 {} {} {:.6f} Anserini\n'.format(id, hits[i].docid, i + 1, hits[i].score))
            cnt += 1
            if cnt % 100 == 0:
                print(f'{cnt} queries completed')


def main():
    print('======================= start running all queries ==============================')
    # different modes to run query expansion experiments
    # nn: nearest neighbors searched with word vectors as expansion
    # wordnet: similar words identified by wordnet as expansion
    # prf: pseudo-relevance feedback to derive expansion
    # rm3: relevance model version 3, a probabilistic language model to derive expansion
    # base: no query expansion
    mode = 'rm3'  # nn, wordnet, prf, rm3, base
    searcher, topics = load_index()
    # k: term threshold, d: doc threshold in prf
    d = 30
    qw = 1
    ks = [2, 5, 10]
    # ks = [1,2,3]
    # ks = [4,5,10]
    # ks = [20,30]
    for k in ks:
        qe = load_qe(mode, num_terms_threshold=k, num_docs_threshold=d, original_query_weight=qw, searcher=searcher)
        run_filename = f'runs/run-msmarco-doc-bm25-{mode}-k{k}-v1.txt'
        if mode == 'prf':
            run_filename = f'runs/run-msmarco-doc-bm25-{mode}-k{k}-d{d}-v1.txt'
        elif mode == 'rm3':
            run_filename = f'runs/run-msmarco-doc-bm25-{mode}-k{k}-d{d}-qw{qw}.txt'
        print('output: ', run_filename)
        run_all_queries(run_filename, topics, searcher, qe)


# main()

from pyserini.search import get_topics
topic_set = 'msmarco_doc_dev'
topics = get_topics(topic_set)
print(f'{len(topics)} topics total')

with open('msmarco-doc-dev-queries.tsv', 'w') as f:
    for query_id in tqdm(topics):
        query = topics[query_id]['title']
        print(str(query_id)+'\t'+query)
        f.write(str(query_id)+'\t'+query+'\n')
