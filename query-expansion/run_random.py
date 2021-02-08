import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"
from tqdm import tqdm
from pyserini.search import SimpleSearcher


def load_qe():
    from qe_random import QE_random
    qe = QE_random()
    return qe


def load_index():
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


def run_all_queries_with(run_filename, topics, searcher, qe):
    with open(run_filename, 'w') as runfile:
        cnt = 0
        print('Running {} queries in total'.format(len(topics)))
        for id in tqdm(topics):
            query = topics[id]['title']
            if qe is not None:
                query = qe.shuffle_query(query)
            hits = searcher.search(query, 1000)
            for i in range(0, len(hits)):
                _ = runfile.write('{} Q0 {} {} {:.6f} Anserini\n'.format(id, hits[i].docid, i+1, hits[i].score))
            cnt += 1
            if cnt % 100 == 0:
                print(f'{cnt} queries completed')


def main():
    print('======================= start running all queries ==============================')
    searcher, topics = load_index()
    qe = load_qe()
    run_filename = f'runs/run-msmarco-doc-bm25-random-try-3.txt'
    print('output: ', run_filename)
    run_all_queries_with(run_filename, topics, searcher, qe)



main()
